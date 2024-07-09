//@ run-pass
use std::process::Command;

fn main() {
    let signalled_version = "Ceci n'est pas une rustc";
    let version = Command::new(std::env::var_os("RUSTC").unwrap())
        .env("RUSTC_FORCE_RUSTC_VERSION", signalled_version)
        .arg("--version")
        .output()
        .unwrap()
        .stdout;
    let version = std::str::from_utf8(&version).unwrap().strip_prefix("rustc ").unwrap().trim_end();
    assert_eq!(version, signalled_version);
}
