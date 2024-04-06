//! Determine rustc version `proc-macro-srv` (and thus the sysroot ABI) is
//! build with and make it accessible at runtime for ABI selection.

use std::{env, fs::File, io::Write, path::PathBuf, process::Command};

fn main() {
    let mut path = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    path.push("rustc_version.rs");
    let mut f = File::create(&path).unwrap();

    let rustc = env::var("RUSTC").expect("proc-macro-srv's build script expects RUSTC to be set");
    let output = Command::new(rustc).arg("--version").output().expect("rustc --version must run");
    let version_string = std::str::from_utf8(&output.stdout[..])
        .expect("rustc --version output must be UTF-8")
        .trim();

    write!(
        f,
        "
    #[allow(dead_code)]
    pub(crate) const RUSTC_VERSION_STRING: &str = {version_string:?};
    "
    )
    .unwrap();
}
