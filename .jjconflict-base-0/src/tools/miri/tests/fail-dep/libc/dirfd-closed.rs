//@ignore-target: windows # no libc
//@compile-flags: -Zmiri-disable-isolation

use std::ffi::CString;
use std::fs::create_dir;
use std::os::unix::ffi::OsStrExt;

#[path = "../../utils/mod.rs"]
mod utils;

#[path = "../../utils/libc.rs"]
mod libc_utils;

use libc_utils::errno_check;

fn main() {
    let path = utils::prepare_dir("miri_test_libc_dirfd_closed");
    create_dir(&path).expect("create_dir failed");
    let cpath = CString::new(path.as_os_str().as_bytes()).expect("CString::new failed");
    let dir: *mut libc::DIR = unsafe { libc::opendir(cpath.as_ptr()) };
    assert!(!dir.is_null());

    let dirfd = unsafe { libc::dirfd(dir) };
    // Mess up the directory stream by closing the backing FD.
    errno_check(unsafe { libc::close(dirfd) });

    errno_check(unsafe { libc::closedir(dir) });
    //~^ERROR: DIR stream has been tampered with
}
