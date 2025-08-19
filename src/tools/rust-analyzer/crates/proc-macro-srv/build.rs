//! Determine rustc version `proc-macro-srv` (and thus the sysroot ABI) is
//! build with and make it accessible at runtime for ABI selection.

use std::{env, process::Command};

fn main() {
    println!("cargo::rustc-check-cfg=cfg(rust_analyzer)");

    let rustc = env::var("RUSTC").expect("proc-macro-srv's build script expects RUSTC to be set");
    #[allow(clippy::disallowed_methods)]
    let output = Command::new(rustc).arg("--version").output().expect("rustc --version must run");
    let version_string = std::str::from_utf8(&output.stdout[..])
        .expect("rustc --version output must be UTF-8")
        .trim();
    println!("cargo::rustc-env=RUSTC_VERSION={version_string}");
}
