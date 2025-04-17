use crate::ffi::{CStr, OsString, c_char};
use crate::os::hermit::ffi::OsStringExt;
use crate::ptr;
use crate::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use crate::sync::atomic::{AtomicIsize, AtomicPtr};

#[path = "common.rs"]
mod common;
pub use common::Args;

static ARGC: AtomicIsize = AtomicIsize::new(0);
static ARGV: AtomicPtr<*const u8> = AtomicPtr::new(ptr::null_mut());

/// One-time global initialization.
pub unsafe fn init(argc: isize, argv: *const *const u8) {
    ARGC.store(argc, Relaxed);
    // Use release ordering here to broadcast writes by the OS.
    ARGV.store(argv as *mut *const u8, Release);
}

/// Returns the command line arguments
pub fn args() -> Args {
    // Synchronize with the store above.
    let argv = ARGV.load(Acquire);
    // If argv has not been initialized yet, do not return any arguments.
    let argc = if argv.is_null() { 0 } else { ARGC.load(Relaxed) };
    let args: Vec<OsString> = (0..argc)
        .map(|i| unsafe {
            let cstr = CStr::from_ptr(*argv.offset(i) as *const c_char);
            OsStringExt::from_vec(cstr.to_bytes().to_vec())
        })
        .collect();

    Args::new(args)
}
