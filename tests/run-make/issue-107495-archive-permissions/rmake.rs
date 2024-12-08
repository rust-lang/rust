#![feature(rustc_private)]

#[cfg(unix)]
extern crate libc;

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use run_make_support::{aux_build, rfs};

fn main() {
    #[cfg(unix)]
    unsafe {
        libc::umask(0o002);
    }

    aux_build().arg("foo.rs").run();
    verify(Path::new("libfoo.rlib"));
}

fn verify(path: &Path) {
    let perm = rfs::metadata(path).permissions();

    assert!(!perm.readonly());

    // Check that the file is readable for everyone
    #[cfg(unix)]
    assert_eq!(perm.mode(), 0o100664);
}
