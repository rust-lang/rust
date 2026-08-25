pub use super::common::Args;
use crate::ffi::OsString;

pub fn args() -> Args {
    let motor_args: Vec<String> = moto_rt::process::args();
    let mut rust_args = Vec::new();

    for arg in motor_args {
        rust_args.push(OsString::from(arg));
    }

    Args::new(rust_args)
}
