//! Architecture-specific support for x86-32 with SSE2 (i686) and x86-64.

mod detect;
mod fma;
mod sqrt;

pub use fma::{fma, fmaf};
pub use sqrt::{sqrt, sqrtf};
