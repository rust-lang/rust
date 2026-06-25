//! Architecture-specific support for LoongArch with FPU

mod fma;
mod rounding;
mod sqrt;

#[cfg(target_feature = "d")]
pub use fma::fma;
#[cfg(target_feature = "f")]
pub use fma::fmaf;
#[cfg(target_feature = "lsx")]
pub use rounding::{rint, rintf};
#[cfg(target_feature = "d")]
pub use sqrt::sqrt;
#[cfg(target_feature = "f")]
pub use sqrt::sqrtf;
