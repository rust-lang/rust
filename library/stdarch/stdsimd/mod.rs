#[unstable(feature = "stdsimd", issue = "0")]
pub mod arch {
    pub use coresimd::arch::*;
    pub mod detect;
}

#[unstable(feature = "stdsimd", issue = "0")]
pub use coresimd::simd;
