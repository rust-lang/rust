// run-pass
// ignore-emscripten no processes
// ignore-sgx no processes

use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    // Checks the behavior of current_dir when used with a relative exe path.
    let me = env::current_exe().unwrap();
    if matches!(env::args().skip(1).next().as_deref(), Some("current-dir")) {
        let cwd = env::current_dir().unwrap();
        assert_eq!(cwd.file_name().unwrap(), "bar");
        std::process::exit(0);
    }
    let exe = me.file_name().unwrap();
    let cwd = std::env::current_dir().unwrap();
    eprintln!("cwd={:?}", cwd);
    let foo = cwd.join("foo");
    let bar = cwd.join("bar");
    fs::create_dir_all(&foo).unwrap();
    fs::create_dir_all(&bar).unwrap();
    fs::copy(&me, foo.join(exe)).unwrap();

    // Unfortunately this is inconsistent based on the platform, see
    // https://github.com/rust-lang/rust/issues/37868. On Windows,
    // it is relative *before* changing the directory, and on Unix
    // it is *after* changing the directory.
    let relative_exe = if cfg!(windows) {
        Path::new("foo").join(exe)
    } else {
        Path::new("../foo").join(exe)
    };

    let status = Command::new(relative_exe)
        .arg("current-dir")
        .current_dir("bar")
        .status()
        .unwrap();
    assert!(status.success());
}
