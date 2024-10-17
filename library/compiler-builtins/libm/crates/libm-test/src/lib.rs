mod num_traits;
mod test_traits;

pub use num_traits::{Float, Hex, Int};
pub use test_traits::{CheckBasis, CheckCtx, CheckOutput, GenerateInput, TupleCall};

/// Result type for tests is usually from `anyhow`. Most times there is no success value to
/// propagate.
pub type TestResult<T = (), E = anyhow::Error> = Result<T, E>;

// List of all files present in libm's source
include!(concat!(env!("OUT_DIR"), "/all_files.rs"));
