#![feature(rustc_private)]

#[cfg(unix)]
extern crate libc;

use run_make_support::{aux_build, fs_wrapper};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

fn main() {
    #[cfg(unix)]
    unsafe {
        libc::umask(0o002);
    }

    aux_build().arg("foo.rs").run();
    verify(Path::new("libfoo.rlib"));
}

fn verify(path: &Path) {
    let perm = fs_wrapper::metadata(path).permissions();

    assert!(!perm.readonly());

    // Check that the file is readable for everyone
    #[cfg(unix)]
    assert_eq!(perm.mode(), 0o100664);
}
