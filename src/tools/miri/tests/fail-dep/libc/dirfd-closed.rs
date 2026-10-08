//@ignore-target: windows # no libc
//@compile-flags: -Zmiri-disable-isolation

use std::fs::create_dir;

#[path = "../../utils/mod.rs"]
mod utils;

#[path = "../../utils/libc.rs"]
mod libc_utils;

use libc_utils::errno_check;

fn main() {
    let path = utils::prepare_dir("miri_test_libc_dirfd_closed");
    create_dir(&path).expect("create_dir failed");
    let cpath = utils::into_c_string(path);
    let dir: *mut libc::DIR = unsafe { libc::opendir(cpath.as_ptr()) };
    assert!(!dir.is_null());

    let dirfd = unsafe { libc::dirfd(dir) };
    // Mess up the directory stream by closing the backing FD.
    errno_check(unsafe { libc::close(dirfd) });

    errno_check(unsafe { libc::closedir(dir) });
    //~^ERROR: DIR stream has been tampered with
}
