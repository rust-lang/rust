// Test that windows verbatim paths in `Command::current_dir` are converted to
// non-verbatim paths before passing to the subprocess.

//@ run-pass
//@ only-windows
//@ needs-subprocess

use std::env;
use std::process::Command;

fn main() {
    if env::args().skip(1).any(|s| s == "--child") {
        child();
    } else {
        parent();
    }
}

fn parent() {
    let exe = env::current_exe().unwrap();
    let dir = env::current_dir().unwrap();
    let status = Command::new(&exe)
        .arg("--child")
        .current_dir(dir.canonicalize().unwrap())
        .spawn()
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(status.code(), Some(0));
}

fn child() {
    let current_dir = env::current_dir().unwrap();
    let current_dir = current_dir.as_os_str().as_encoded_bytes();
    assert!(!current_dir.starts_with(br"\\?\"));
}
