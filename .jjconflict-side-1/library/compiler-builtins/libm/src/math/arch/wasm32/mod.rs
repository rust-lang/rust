//! Wasm has builtins for simple float operations. Use the unstable `core::arch` intrinsics which
//! are significantly faster than soft float operations.

mod fabs;
mod rounding;
mod sqrt;

pub use fabs::{fabs, fabsf};
pub use rounding::{ceil, ceilf, floor, floorf, rint, rintf, trunc, truncf};
pub use sqrt::{sqrt, sqrtf};
