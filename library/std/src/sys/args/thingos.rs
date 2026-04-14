//! ThingOS command-line argument collection.
//!
//! Argc/argv are stored during PAL initialisation (via `init`) and returned
//! on demand through `args()`.

pub use super::common::Args;
use crate::ffi::{CStr, OsString};
use crate::sync::atomic::{AtomicIsize, AtomicPtr, Ordering};

static ARGC: AtomicIsize = AtomicIsize::new(0);
static ARGV: AtomicPtr<*const u8> = AtomicPtr::new(core::ptr::null_mut());

/// Called once from `sys::pal::thingos::init()`.
///
/// # Safety
/// `argc` and `argv` must be valid for the entire lifetime of the process.
pub unsafe fn init(argc: isize, argv: *const *const u8) {
    ARGC.store(argc, Ordering::Relaxed);
    ARGV.store(argv as *mut _, Ordering::Relaxed);
}

/// Returns the collected command-line arguments.
pub fn args() -> Args {
    let argc = ARGC.load(Ordering::Relaxed);
    let argv = ARGV.load(Ordering::Relaxed) as *const *const u8;

    let mut vec = Vec::with_capacity(argc as usize);
    for i in 0..argc {
        // SAFETY: argv is valid for `argc` pointers as guaranteed by `init`.
        let ptr = unsafe { argv.offset(i).read() };
        if ptr.is_null() {
            break;
        }
        // SAFETY: Each argv[i] is a valid NUL-terminated C string.
        let cstr = unsafe { CStr::from_ptr(ptr) };
        vec.push(OsString::from(
            core::str::from_utf8(cstr.to_bytes()).unwrap_or_default(),
        ));
    }
    Args::new(vec)
}
