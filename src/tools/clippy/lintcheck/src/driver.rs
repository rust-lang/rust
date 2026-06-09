use crate::recursive::{DriverInfo, deserialize_line, serialize_line};

use std::io::{self, BufReader, Write};
use std::net::TcpStream;
use std::process::{self, Command, Stdio};
use std::{env, mem};

/// 1. Sends [`DriverInfo`] to the [`crate::recursive::LintcheckServer`] running on `addr`
/// 2. Receives [bool] from the server, if `false` returns `None`
/// 3. Otherwise sends the stderr of running `clippy-driver` to the server
fn run_clippy(addr: &str) -> Option<i32> {
    let driver_info = DriverInfo {
        package_name: env::var("CARGO_PKG_NAME").ok()?,
        version: env::var("CARGO_PKG_VERSION").ok()?,
    };

    let mut stream = BufReader::new(TcpStream::connect(addr).unwrap());

    serialize_line(&driver_info, stream.get_mut());

    let should_run = deserialize_line::<bool, _>(&mut stream);
    if !should_run {
        return None;
    }

    // Remove --cap-lints allow so that clippy runs and lints are emitted
    let mut include_next = true;
    let args = env::args().skip(1).filter(|arg| match arg.as_str() {
        "--cap-lints=allow" => false,
        "--cap-lints" => {
            include_next = false;
            false
        },
        _ => mem::replace(&mut include_next, true),
    });

    let output = Command::new(env::var("CLIPPY_DRIVER").expect("missing env CLIPPY_DRIVER"))
        .args(args)
        .stdout(Stdio::inherit())
        .output()
        .expect("failed to run clippy-driver");

    stream
        .get_mut()
        .write_all(&output.stderr)
        .unwrap_or_else(|e| panic!("{e:?} in {driver_info:?}"));

    match output.status.code() {
        Some(0) => Some(0),
        code => {
            io::stderr().write_all(&output.stderr).unwrap();
            Some(code.expect("killed by signal"))
        },
    }
}

pub fn drive(addr: &str) {
    process::exit(run_clippy(addr).unwrap_or_else(|| {
        Command::new("rustc")
            .args(env::args_os().skip(2))
            .status()
            .unwrap()
            .code()
            .unwrap()
    }))
}
