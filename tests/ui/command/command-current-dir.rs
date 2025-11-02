//@ run-pass
//@ no-prefer-dynamic We move the binary around, so do not depend dynamically on libstd
//@ needs-subprocess
//@ ignore-fuchsia Needs directory creation privilege
//@ ignore-tvos `Command::current_dir` requires fork, which is prohibited
//@ ignore-watchos `Command::current_dir` requires fork, which is prohibited

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
    let cwd = me.parent().unwrap();
    eprintln!("cwd={:?}", cwd);
    // Change directory to where the executable is located, since this test
    // fundamentally needs to use relative paths. In some cases (like
    // remote-test-server), the current_dir can be somewhere else, so make
    // sure it is something we can use. We assume we can write to this
    // directory.
    env::set_current_dir(&cwd).unwrap();
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
