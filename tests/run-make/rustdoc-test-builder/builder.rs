use std::ffi::OsString;
use std::path::PathBuf;
use std::process::{self, Command};
use std::{env, fs};

fn main() {
    let args: Vec<OsString> = env::args_os().collect();
    let log_path = env::var_os("BUILDER_LOG").map(PathBuf::from).expect("BUILDER_LOG must be set");
    let real_rustc = env::var_os("REAL_RUSTC").expect("REAL_RUSTC must be set");

    let log_contents =
        args.iter().skip(1).map(|arg| arg.to_string_lossy()).collect::<Vec<_>>().join("\n");
    fs::write(&log_path, log_contents).expect("failed to write builder log");

    let status = Command::new(real_rustc)
        .args(args.iter().skip(1))
        .status()
        .expect("failed to invoke real rustc");

    if !status.success() {
        process::exit(status.code().unwrap_or(1));
    }
}
