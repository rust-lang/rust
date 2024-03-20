extern crate run_make_support;

use run_make_support::{aux_build, out_dir};
use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

fn main() {
    aux_build().arg("foo.rs").run();
    verify(&out_dir().join("libfoo.rlib"));
}

fn verify(path: &Path) {
    let perm = fs::metadata(path).unwrap().permissions();

    assert!(!perm.readonly());

    // Check that the file is readable for everyone
    #[cfg(unix)]
    assert_eq!(perm.mode(), 0o100664);
}
