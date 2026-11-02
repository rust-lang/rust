// For some reason on Windows, the PATH to the libstd dylib doesn't seem to
// carry over to running the runtool.
//@ no-prefer-dynamic

use std::path::Path;
use std::process::Command;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    eprintln!("{args:#?}");
    assert_eq!(args.len(), 4);
    assert_eq!(args[1], "arg1");
    assert_eq!(args[2], "arg2 with space");
    let path = Path::new(&args[3]);
    let output = Command::new(path).output().unwrap();
    // Should fail without env var.
    assert!(!output.status.success());
    let output = Command::new(path).env("DOCTEST_RUNTOOL_CHECK", "xyz").output().unwrap();
    // Should pass with env var.
    assert!(output.status.success());
}
