pub mod gen;
#[cfg(feature = "test-multiprecision")]
pub mod mpfloat;
pub mod op;
mod precision;
mod test_traits;

pub use libm::support::{Float, Int};
pub use op::{BaseName, MathOp, Name};
pub use precision::{MaybeOverride, SpecialCase, multiprec_allowed_ulp, musl_allowed_ulp};
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
