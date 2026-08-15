//! Architecture-specific support for aarch64 with neon.

mod fma;
mod rounding;
mod sqrt;

pub use fma::{fma, fmaf};
#[cfg(all(f16_enabled, target_feature = "fp16"))]
pub use rounding::rintf16;
pub use rounding::{rint, rintf};
#[cfg(all(f16_enabled, target_feature = "fp16"))]
pub use sqrt::sqrtf16;
pub use sqrt::{sqrt, sqrtf};
