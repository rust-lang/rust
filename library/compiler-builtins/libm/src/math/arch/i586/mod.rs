//! Architecture-specific support for x86-32 without SSE2
//!
//! We use an alternative implementation on x86, because the
//! main implementation fails with the x87 FPU used by
//! debian i386, probably due to excess precision issues.
//!
//! See https://github.com/rust-lang/compiler-builtins/pull/976 for discussion on why these
//! functions are implemented in this way.

mod exp_all;
mod rounding;

pub use exp_all::{x87_exp, x87_exp2, x87_exp2f, x87_exp10, x87_exp10f, x87_expf};
pub use rounding::{ceil, floor};
