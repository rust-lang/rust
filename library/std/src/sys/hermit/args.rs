use crate::ffi::{CStr, OsString};
use crate::os::unix::ffi::OsStringExt;
use crate::ptr;
use crate::sys_common::mutex::StaticMutex;
use crate::vec::IntoIter;

static mut ARGC: isize = 0;
static mut ARGV: *const *const u8 = ptr::null();
static LOCK: StaticMutex = StaticMutex::new();

/// One-time global initialization.
pub unsafe fn init(argc: isize, argv: *const *const u8) {
    let _guard = LOCK.lock();
    ARGC = argc;
    ARGV = argv;
}

/// One-time global cleanup.
pub unsafe fn cleanup() {
    let _guard = LOCK.lock();
    ARGC = 0;
    ARGV = ptr::null();
}

/// Returns the command line arguments
pub fn args() -> Args {
    unsafe {
        let _guard = LOCK.lock();
        (0..ARGC)
            .map(|i| {
                let cstr = CStr::from_ptr(*ARGV.offset(i) as *const i8);
                OsStringExt::from_vec(cstr.to_bytes().to_vec())
            })
            .collect::<Vec<_>>()
            .into_iter()
    }
}

pub type Args = IntoIter<OsString>;
