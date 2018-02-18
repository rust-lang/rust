pub mod arch {
    pub use coresimd::arch::*;
    pub mod detect;
}

pub use coresimd::simd;
