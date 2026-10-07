//@ignore-target: windows # no libc
//@compile-flags: -Zmiri-disable-isolation

use std::fs::create_dir;

#[path = "../../utils/mod.rs"]
mod utils;

#[path = "../../utils/libc.rs"]
mod libc_utils;

use libc_utils::{errno_check, errno_result};

fn main() {
    let path = utils::prepare("miri_test_libc_dirfd_replaced");
    create_dir(&path).expect("create_dir failed");
    let cpath = utils::into_c_string(path);
    let dir: *mut libc::DIR = unsafe { libc::opendir(cpath.as_ptr()) };
    assert!(!dir.is_null());

    let dirfd = unsafe { libc::dirfd(dir) };
    // Mess up the directory stream by replacing the backing FD.
    errno_result(unsafe { libc::dup2(0, dirfd) }).unwrap();

    let _entry_ptr = unsafe { libc::readdir(dir) };
    //~^ERROR: DIR stream has been tampered with

    errno_check(unsafe { libc::closedir(dir) });
}
