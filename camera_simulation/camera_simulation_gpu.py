import math
import cupy as cp


aperture_values = [5.0, 9.0, 16.0]
shutter_speed_values = [1/4, 1/60, 1/1000]
shutter_speed_values_str = ["1/4", "1/60", "1/1000"]
iso_values = [250, 2000, 16000]


class CameraSimulation:
    # ISO 250 / 2000 / 16000
    # Shutter Speed (1/4) / (1/60) / (1/1000)
    # Aperture f5 / f9 / f16

    # Camera Parameters from Basler acA1920-155um

    def __init__(
        self,
        iso=2000,
        shutter_speed=1/60,
        aperture=9,
        log=True,
        camera_type="alpha6000"
    ):
        self.iso = iso
        self.shutter_speed = shutter_speed
        self.aperture = aperture
        self.iso_factor = iso / 100
        self.log = log

        if camera_type == "alpha6000":
            self.qe = 0.5
            self.inverse_K = 0.425**-1
            self.dark_noise_sigma = 2.43
            self.saturation_capacity = 9091
            self.abs_sensitivity_threshold = 10
        else:
            self.qe = 0.7
            self.inverse_K = 8.4
            self.dark_noise_sigma = 6.8
            self.saturation_capacity = 32700
            self.abs_sensitivity_threshold = 10

    def set_iso(self, iso):
        self.iso = iso
        self.iso_factor = iso / 100

    def set_shutter_speed(self, shutter_speed):
        self.shutter_speed = shutter_speed

    def set_aperture(self, aperture):
        self.aperture = aperture

    def set_log(self, log):
        self.log = log

    def set_parameters(self, iso, shutter_speed, aperture):
        self.set_iso(iso)
        self.set_shutter_speed(shutter_speed)
        self.set_aperture(aperture)

    def get_parameters(self):
        return {
            "iso": self.iso,
            "shutter_speed": self.shutter_speed,
            "aperture": self.aperture
        }

    def log_image_stats(self, image):
        if self.log:
            print(f"Image shape: {image.shape}")
            print(f"Image dtype: {image.dtype}")
            print(f"Pixel value range: {cp.min(image).get()} to {cp.max(image).get()}")
            print(f"Mean pixel value: {cp.mean(image).get()}")
            print(f"Standard deviation of pixel values: {cp.std(image).get()}")

    def input_to_illuminance(self, image):
        # Work entirely on GPU
        min_value = cp.min(image)

        if min_value < 0:
            image = image - min_value

        illuminance = (
            (math.pi * image) / 4.0
            * (self.aperture ** 2)
        )

        return illuminance

    def illuminance_to_photons_with_shot_noise(self, illuminance):
        # GPU Poisson noise
        expected_photons = illuminance * self.shutter_speed

        photons = cp.random.poisson(expected_photons)

        return photons

    def photons_to_electrons(self, photons):
        return photons * self.qe

    def apply_iso(self, x):
        return x * self.iso_factor

    def apply_system_gain(self, x):
        return x / self.inverse_K

    def add_dark_noise(self, x):
        # GPU Gaussian noise
        read_noise = cp.random.normal(
            loc=0.0,
            scale=math.sqrt(self.dark_noise_sigma),
            size=x.shape
        )

        return x + read_noise

    def clip_electrons(self, x):
        return cp.clip(
            x,
            0,
            self.saturation_capacity
        )

    def quantize_to_8bit(self, x):
        return (
            x / self.saturation_capacity * 255
        ).astype(cp.uint8)

    def simulate_image(self, image, depth_map=None):
        # Move image to GPU once
        return_to_cpu = False
        if not isinstance(image, cp.ndarray):
            return_to_cpu = True
            image = cp.asarray(image)

        self.log_image_stats(image)

        # Input -> Illuminance
        illuminance = self.input_to_illuminance(image)

        # Illuminance -> Photons + Shot Noise
        photons = self.illuminance_to_photons_with_shot_noise(
            illuminance
        )

        # Photons -> Electrons
        x = self.photons_to_electrons(photons)

        self.log_image_stats(x)

        # System gain
        x = self.apply_system_gain(x)

        # ISO
        x = self.apply_iso(x)

        # Dark/read noise
        x = self.add_dark_noise(x)

        # Sensor saturation
        x = self.clip_electrons(x)

        # 8-bit quantization
        x = self.quantize_to_8bit(x)

        self.log_image_stats(x)

        if return_to_cpu:
            x = x.get()

        return x