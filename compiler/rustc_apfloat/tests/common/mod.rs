use rustc_apfloat::ieee::{Double, Single};
use rustc_apfloat::Float;

pub(crate) trait SingleExt {
    fn from_f32(input: f32) -> Self;
    fn to_f32(self) -> f32;
}

impl SingleExt for Single {
    fn from_f32(input: f32) -> Self {
        Self::from_bits(input.to_bits() as u128)
    }

    fn to_f32(self) -> f32 {
        f32::from_bits(self.to_bits() as u32)
    }
}

pub(crate) trait DoubleExt {
    fn from_f64(input: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl DoubleExt for Double {
    fn from_f64(input: f64) -> Self {
        Self::from_bits(input.to_bits() as u128)
    }

    fn to_f64(self) -> f64 {
        f64::from_bits(self.to_bits() as u64)
    }
}
