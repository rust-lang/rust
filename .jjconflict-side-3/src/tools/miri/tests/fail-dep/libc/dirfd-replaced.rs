//@ignore-target: windows # no libc
//@compile-flags: -Zmiri-disable-isolation

use std::ffi::CString;
use std::fs::create_dir;
use std::os::unix::ffi::OsStrExt;

#[path = "../../utils/mod.rs"]
mod utils;

#[path = "../../utils/libc.rs"]
mod libc_utils;

use libc_utils::{errno_check, errno_result};

fn main() {
    let path = utils::prepare_dir("miri_test_libc_dirfd_replaced");
    create_dir(&path).expect("create_dir failed");
    let cpath = CString::new(path.as_os_str().as_bytes()).expect("CString::new failed");
    let dir: *mut libc::DIR = unsafe { libc::opendir(cpath.as_ptr()) };
    assert!(!dir.is_null());

    let dirfd = unsafe { libc::dirfd(dir) };
    // Mess up the directory stream by replacing the backing FD.
    errno_result(unsafe { libc::dup2(0, dirfd) }).unwrap();

    let _entry_ptr = unsafe { libc::readdir(dir) };
    //~^ERROR: DIR stream has been tampered with

    errno_check(unsafe { libc::closedir(dir) });
}
