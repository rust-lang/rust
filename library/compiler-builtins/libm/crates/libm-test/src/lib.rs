#![allow(clippy::unusual_byte_groupings)] // sometimes we group by sign_exp_sig

pub mod domain;
mod f8_impl;
pub mod gen;
#[cfg(feature = "test-multiprecision")]
pub mod mpfloat;
mod num;
pub mod op;
mod precision;
mod run_cfg;
mod test_traits;

pub use f8_impl::f8;
pub use libm::support::{Float, Int, IntTy, MinInt};
pub use num::{FloatExt, logspace};
pub use op::{BaseName, FloatTy, Identifier, MathOp, OpCFn, OpFTy, OpRustFn, OpRustRet, Ty};
pub use precision::{MaybeOverride, SpecialCase, default_ulp};
pub use run_cfg::{CheckBasis, CheckCtx};
pub use test_traits::{CheckOutput, GenerateInput, Hex, TupleCall};

/// Result type for tests is usually from `anyhow`. Most times there is no success value to
/// propagate.
pub type TestResult<T = (), E = anyhow::Error> = Result<T, E>;

/// True if `EMULATED` is set and nonempty. Used to determine how many iterations to run.
pub const fn emulated() -> bool {
    match option_env!("EMULATED") {
        Some(s) if s.is_empty() => false,
        None => false,
        Some(_) => true,
    }
}

/// True if `CI` is set and nonempty.
pub const fn ci() -> bool {
    match option_env!("CI") {
        Some(s) if s.is_empty() => false,
        None => false,
        Some(_) => true,
    }
}
