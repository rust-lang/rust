#![cfg_attr(f16_enabled, feature(f16))]
#![cfg_attr(f128_enabled, feature(f128))]
#![allow(clippy::unusual_byte_groupings)] // sometimes we group by sign_exp_sig

pub mod domain;
mod f8_impl;
pub mod gen;
#[cfg(feature = "build-mpfr")]
pub mod mpfloat;
mod num;
pub mod op;
mod precision;
mod run_cfg;
mod test_traits;

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::LazyLock;
use std::time::SystemTime;

pub use f8_impl::f8;
pub use libm::support::{Float, Int, IntTy, MinInt};
pub use num::{FloatExt, linear_ints, logspace};
pub use op::{
    BaseName, FloatTy, Identifier, MathOp, OpCFn, OpCRet, OpFTy, OpRustArgs, OpRustFn, OpRustRet,
    Ty,
};
pub use precision::{MaybeOverride, SpecialCase, default_ulp};
use run_cfg::EXTENSIVE_MAX_ITERATIONS;
pub use run_cfg::{CheckBasis, CheckCtx, EXTENSIVE_ENV, GeneratorKind, skip_extensive_test};
pub use test_traits::{CheckOutput, Hex, TupleCall};

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

/// Print to stderr and additionally log it to `target/test-log.txt`. This is useful for saving
/// output that would otherwise be consumed by the test harness.
pub fn test_log(s: &str) {
    // Handle to a file opened in append mode, unless a suitable path can't be determined.
    static OUTFILE: LazyLock<Option<File>> = LazyLock::new(|| {
        // If the target directory is overridden, use that environment variable. Otherwise, save
        // at the default path `{workspace_root}/target`.
        let target_dir = match env::var("CARGO_TARGET_DIR") {
            Ok(s) => PathBuf::from(s),
            Err(_) => {
                let Ok(x) = env::var("CARGO_MANIFEST_DIR") else {
                    return None;
                };

                PathBuf::from(x).parent().unwrap().parent().unwrap().join("target")
            }
        };
        let outfile = target_dir.join("test-log.txt");

        let mut f = File::options()
            .create(true)
            .append(true)
            .open(outfile)
            .expect("failed to open logfile");
        let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap();

        writeln!(f, "\n\nTest run at {}", now.as_secs()).unwrap();
        writeln!(f, "arch: {}", env::consts::ARCH).unwrap();
        writeln!(f, "os: {}", env::consts::OS).unwrap();
        writeln!(f, "bits: {}", usize::BITS).unwrap();
        writeln!(f, "emulated: {}", emulated()).unwrap();
        writeln!(f, "ci: {}", ci()).unwrap();
        writeln!(f, "cargo features: {}", env!("CFG_CARGO_FEATURES")).unwrap();
        writeln!(f, "opt level: {}", env!("CFG_OPT_LEVEL")).unwrap();
        writeln!(f, "target features: {}", env!("CFG_TARGET_FEATURES")).unwrap();
        writeln!(f, "extensive iterations {}", *EXTENSIVE_MAX_ITERATIONS).unwrap();

        Some(f)
    });

    eprintln!("{s}");

    if let Some(mut f) = OUTFILE.as_ref() {
        writeln!(f, "{s}").unwrap();
    }
}
