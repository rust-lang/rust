#![allow(clippy::unusual_byte_groupings)] // sometimes we group by sign_exp_sig

mod f8_impl;
pub mod gen;
#[cfg(feature = "test-multiprecision")]
pub mod mpfloat;
pub mod op;
mod precision;
mod test_traits;

pub use f8_impl::f8;
pub use libm::support::{Float, Int, IntTy};
pub use op::{BaseName, Identifier, MathOp, OpCFn, OpFTy, OpRustFn, OpRustRet};
pub use precision::{MaybeOverride, SpecialCase, default_ulp};
pub use test_traits::{CheckBasis, CheckCtx, CheckOutput, GenerateInput, Hex, TupleCall};

/// Result type for tests is usually from `anyhow`. Most times there is no success value to
/// propagate.
pub type TestResult<T = (), E = anyhow::Error> = Result<T, E>;

// List of all files present in libm's source
include!(concat!(env!("OUT_DIR"), "/all_files.rs"));

/// True if `EMULATED` is set and nonempty. Used to determine how many iterations to run.
pub const fn emulated() -> bool {
    match option_env!("EMULATED") {
        Some(s) if s.is_empty() => false,
        None => false,
        Some(_) => true,
    }
}
