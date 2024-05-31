import cv2

import torch
from torchvision import transforms
# from network.R3D import R3D
# from network.R2Plus1D import R2Plus1D
# from network.T3D import D3D, T3D
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial


# from torchsummary import summary

def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)

        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block,  # BasicBlock or Bottleneck
            layers,  # number of blocks for each layer
            block_inplanes,  # number of input channels for each layer
            n_input_channels=3,  # number of input channels
            conv1_t_size=7,  # kernel size in t for the first conv layer
            conv1_t_stride=1,  # stride in t for the first conv layer
            no_max_pool=False,  # whether to use max pool
            widen_factor=1.0,  # widen factor
            n_classes=27  # number of classes
    ):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        # First convolution
        self.conv1 = nn.Conv3d(
            n_input_channels,
            self.in_planes,
            kernel_size=(conv1_t_size, 7, 7),
            stride=(conv1_t_stride, 2, 2),
            padding=(conv1_t_size // 2, 3, 3),
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        # Layer 1
        self.layer1 = self._make_layer(
            block,
            block_inplanes[0],
            layers[0],
        )
        # Layer 2
        self.layer2 = self._make_layer(
            block,
            block_inplanes[1],
            layers[1],
            stride=2
        )
        # Layer 3
        self.layer3 = self._make_layer(
            block,
            block_inplanes[2],
            layers[2],
            stride=2
        )
        # Layer 4
        self.layer4 = self._make_layer(
            block,
            block_inplanes[3],
            layers[3],
            stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # Downsample input x and zero padding before adding it with out (BasicBlock and Bottleneck)
    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(
            out.size(0), planes - out.size(1),
            out.size(2), out.size(3), out.size(4)
        )
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = partial(
                self._downsample_basic_block,
                planes=planes * block.expansion,
                stride=stride
            )

        layers = []
        layers.append(
            block(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
                downsample=downsample
            )
        )
        self.in_planes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def R3D(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152]

    if model_depth == 10:
        model = ResNet(
            BasicBlock,  # block
            [1, 1, 1, 1],  # layers
            get_inplanes(),  # block_inplanes: [64, 128, 256, 512]
            **kwargs  # others
        )
    elif model_depth == 18:
        model = ResNet(
            BasicBlock,
            [2, 2, 2, 2],
            get_inplanes(),
            **kwargs
        )
    elif model_depth == 34:
        model = ResNet(
            BasicBlock,
            [3, 4, 6, 3],
            get_inplanes(),
            **kwargs
        )
    elif model_depth == 50:
        model = ResNet(
            Bottleneck,
            [3, 4, 6, 3],
            get_inplanes(),
            **kwargs
        )
    elif model_depth == 101:
        model = ResNet(
            Bottleneck,
            [3, 4, 23, 3],
            get_inplanes(),
            **kwargs
        )
    elif model_depth == 152:
        model = ResNet(
            Bottleneck,
            [3, 8, 36, 3],
            get_inplanes(),
            **kwargs
        )

    return model




class GestureRecognizer:
    def __init__(
            self,
            model_path,
            model_arch='r3d', block_arch=121,
            resize=(112, 112), num_frames=30,
            no_max_pool=True, n_classes=27,
            drop_frame=0
    ):
        # Model path
        self.model_path = model_path
        # Model parameters
        self.model_arch = model_arch
        self.block_arch = block_arch
        self.resize = resize
        self.num_frames = num_frames
        self.no_max_pool = no_max_pool
        self.n_classes = n_classes
        # Drop n frames between 2 frames
        self.drop_frame = drop_frame

    def load_model(self, device):
        if self.model_arch == 'r3d':
            model = R3D(
                self.block_arch,
                n_input_channels=3,
                conv1_t_size=7,
                conv1_t_stride=1,
                no_max_pool=self.no_max_pool,
                widen_factor=1,
                n_classes=self.n_classes
            ).to(device)
        elif self.model_arch == 'r2plus1d':
            model = R2Plus1D(
                self.block_arch,
                n_input_channels=3,
                conv1_t_size=7,
                conv1_t_stride=1,
                no_max_pool=self.no_max_pool,
                widen_factor=1,
                n_classes=self.n_classes
            ).to(device)
        elif self.model_arch == 't3d':
            model = T3D(
                self.block_arch,
                phi=0.5,
                growth_rate=12,
                temporal_expansion=1,
                transition_t1_size=[1, 3, 6],
                transition_t_size=[1, 3, 4],
                n_input_channels=3,
                conv1_t_size=3,
                conv1_t_stride=1,
                no_max_pool=self.no_max_pool,
                n_classes=self.n_classes,
                dropout=0.0
            )
        elif self.model_arch == 'd3d':
            model = D3D(
                self.block_arch,
                phi=0.5,
                growth_rate=12,
                n_input_channels=3,
                conv1_t_size=3,
                conv1_t_stride=1,
                no_max_pool=self.no_max_pool,
                n_classes=self.n_classes,
                dropout=0.0
            ).to(device)
        else:
            raise ValueError('Model architecture not supported')

        # Load model
        model.load_state_dict(torch.load(self.model_path, map_location=device))

        return model

    def run(self):
        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.resize),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Choose device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        # Load model
        model = self.load_model(device)

        # Use the default camera as the video source
        cap = cv2.VideoCapture(0)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        count = 0
        frames = []
        while True:
            # Update count
            count += 1
            # Drop frames
            if count % (self.drop_frame + 1) != 0:
                continue

            # Read frame
            _, frame = cap.read()
            if frame is None:
                break
            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)

            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Apply transformations
            frame_tensor = transform(frame_rgb)
            frames.append(frame_tensor)

            # If enough frames are collected, make a prediction
            if len(frames) == self.num_frames:
                # Stack frames along the time dimension
                input_frames = torch.stack(frames, dim=0)  # (T x C x H x W)
                input_frames = input_frames.permute(1, 0, 2, 3)  # (C x T x H x W)
                input_frames = input_frames.unsqueeze(0)  # Add batch dimension
                input_frames = input_frames.to(device)

                with torch.no_grad():
                    # Set model to evaluation mode
                    model.eval()
                    # Forward pass
                    output = model(input_frames)
                    # Get prediction
                    _, pred = output.max(1)
                    # Get the correspoding label name
                    predicted_label = labels[pred.item()]

                print(pred, predicted_label)

                # Display the result on the frame at the bottom with red color
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottom_left_corner_of_text = (5, frame_height - 10)  # Adjust the Y-coordinate for the bottom
                font_scale = 0.25
                font_color = (0, 0, 255)  # Red color in BGR format
                font_thickness = 1
                cv2.putText(
                    frame,
                    f'{predicted_label} ({pred.item()})',
                    bottom_left_corner_of_text, font, font_scale, font_color, font_thickness,
                    cv2.LINE_AA
                )

                # Remove the first frame
                frames = frames[1:]

            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to stop recording
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Release the video capture object
        cap.release()
        # Close all the frames
        cv2.destroyAllWindows()


def main():
    program = GestureRecognizer(
        model_path='r3d-18_0-mp_3-epochs.pth',
        model_arch='r3d', block_arch=18,
        drop_frame=0,
        n_classes=27
    )

    program.run()


if __name__ == '__main__':
    labels = [
        'Swiping Left',
        'Swiping Right',
        'Swiping Down',
        'Swiping Up',
        'Pushing Hand Away',
        'Pulling Hand In',
        'Sliding Two Fingers Left',
        'Sliding Two Fingers Right',
        'Sliding Two Fingers Down',
        'Sliding Two Fingers Up',
        'Pushing Two Fingers Away',
        'Pulling Two Fingers In',
        'Rolling Hand Forward',
        'Rolling Hand Backward',
        'Turning Hand Clockwise',
        'Turning Hand Counterclockwise',
        'Zooming In With Full Hand',
        'Zooming Out With Full Hand',
        'Zooming In With Two Fingers',
        'Zooming Out With Two Fingers',
        'Thumb Up',
        'Thumb Down',
        'Shaking Hand',
        'Stop Sign',
        'Drumming Fingers',
        'No gesture',
        'Doing other things',
    ]

    main()
