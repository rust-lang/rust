//@compile-flags: -Zmiri-disable-isolation
//@ignore-target: windows # shim not supported
//@ignore-host: windows # needs unix PermissionExt

use std::fs::{self, File};

#[path = "../../utils/mod.rs"]
mod utils;

macro_rules! check {
    ($e:expr) => {
        match $e {
            Ok(t) => t,
            Err(e) => panic!("{} failed with: {e}", stringify!($e)),
        }
    };
}

fn main() {
    chmod_works();
    fchmod_works();
}

fn chmod_works() {
    let tmpdir = utils::tmp();
    let file = tmpdir.join("miri_test_fs_set_permissions.txt");

    check!(File::create(&file));
    let attr = check!(fs::metadata(&file));
    assert!(!attr.permissions().readonly());
    let mut p = attr.permissions();
    p.set_readonly(true);
    check!(fs::set_permissions(&file, p.clone()));
    let attr = check!(fs::metadata(&file));
    assert!(attr.permissions().readonly());

    match fs::set_permissions(&tmpdir.join("foo"), p.clone()) {
        Ok(..) => panic!("wanted an error"),
        Err(..) => {}
    }

    p.set_readonly(false);
    check!(fs::set_permissions(&file, p));
}

fn fchmod_works() {
    let tmpdir = utils::tmp();
    let path = tmpdir.join("miri_test_file_set_permissions.txt");

    let file = check!(File::create(&path));
    let attr = check!(fs::metadata(&path));
    assert!(!attr.permissions().readonly());
    let mut p = attr.permissions();
    p.set_readonly(true);
    check!(file.set_permissions(p.clone()));
    let attr = check!(fs::metadata(&path));
    assert!(attr.permissions().readonly());

    p.set_readonly(false);
    check!(file.set_permissions(p));
}
