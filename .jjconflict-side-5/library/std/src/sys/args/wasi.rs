#[cfg(target_env = "p2")]
use wasip2 as wasi;
#[cfg(target_env = "p3")]
use wasip3 as wasi;

pub use super::common::Args;

/// Returns the command line arguments
pub fn args() -> Args {
    Args::new(wasi::cli::environment::get_arguments().into_iter().map(|arg| arg.into()).collect())
}
