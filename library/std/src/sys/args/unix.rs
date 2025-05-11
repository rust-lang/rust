//! Global initialization and retrieval of command line arguments.
//!
//! On some platforms these are stored during runtime startup,
//! and on some they are retrieved from the system on demand.

#![allow(dead_code)] // runtime init functions not used during testing

pub use super::common::Args;
use crate::ffi::CStr;
#[cfg(target_os = "hermit")]
use crate::os::hermit::ffi::OsStringExt;
#[cfg(not(target_os = "hermit"))]
use crate::os::unix::ffi::OsStringExt;

/// One-time global initialization.
pub unsafe fn init(argc: isize, argv: *const *const u8) {
    unsafe { imp::init(argc, argv) }
}

/// Returns the command line arguments
pub fn args() -> Args {
    let (argc, argv) = imp::argc_argv();

    let mut vec = Vec::with_capacity(argc as usize);

    for i in 0..argc {
        // SAFETY: `argv` is non-null if `argc` is positive, and it is
        // guaranteed to be at least as long as `argc`, so reading from it
        // should be safe.
        let ptr = unsafe { argv.offset(i).read() };

        // Some C commandline parsers (e.g. GLib and Qt) are replacing already
        // handled arguments in `argv` with `NULL` and move them to the end.
        //
        // Since they can't directly ensure updates to `argc` as well, this
        // means that `argc` might be bigger than the actual number of
        // non-`NULL` pointers in `argv` at this point.
        //
        // To handle this we simply stop iterating at the first `NULL`
        // argument. `argv` is also guaranteed to be `NULL`-terminated so any
        // non-`NULL` arguments after the first `NULL` can safely be ignored.
        if ptr.is_null() {
            // NOTE: On Apple platforms, `-[NSProcessInfo arguments]` does not
            // stop iterating here, but instead `continue`, always iterating
            // up until it reached `argc`.
            //
            // This difference will only matter in very specific circumstances
            // where `argc`/`argv` have been modified, but in unexpected ways,
            // so it likely doesn't really matter which option we choose.
            // See the following PR for further discussion:
            // <https://github.com/rust-lang/rust/pull/125225>
            break;
        }

        // SAFETY: Just checked that the pointer is not NULL, and arguments
        // are otherwise guaranteed to be valid C strings.
        let cstr = unsafe { CStr::from_ptr(ptr) };
        vec.push(OsStringExt::from_vec(cstr.to_bytes().to_vec()));
    }

    Args::new(vec)
}

#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "freebsd",
    target_os = "dragonfly",
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "cygwin",
    target_os = "solaris",
    target_os = "illumos",
    target_os = "emscripten",
    target_os = "haiku",
    target_os = "hermit",
    target_os = "l4re",
    target_os = "fuchsia",
    target_os = "redox",
    target_os = "vxworks",
    target_os = "horizon",
    target_os = "aix",
    target_os = "nto",
    target_os = "hurd",
    target_os = "rtems",
    target_os = "nuttx",
))]
mod imp {
    use crate::ffi::c_char;
    use crate::ptr;
    use crate::sync::atomic::{Atomic, AtomicIsize, AtomicPtr, Ordering};

    // The system-provided argc and argv, which we store in static memory
    // here so that we can defer the work of parsing them until its actually
    // needed.
    //
    // Note that we never mutate argv/argc, the argv array, or the argv
    // strings, which allows the code in this file to be very simple.
    static ARGC: Atomic<isize> = AtomicIsize::new(0);
    static ARGV: Atomic<*mut *const u8> = AtomicPtr::new(ptr::null_mut());

    unsafe fn really_init(argc: isize, argv: *const *const u8) {
        // These don't need to be ordered with each other or other stores,
        // because they only hold the unmodified system-provided argv/argc.
        ARGC.store(argc, Ordering::Relaxed);
        ARGV.store(argv as *mut _, Ordering::Relaxed);
    }

    #[inline(always)]
    pub unsafe fn init(argc: isize, argv: *const *const u8) {
        // on GNU/Linux if we are main then we will init argv and argc twice, it "duplicates work"
        // BUT edge-cases are real: only using .init_array can break most emulators, dlopen, etc.
        unsafe { really_init(argc, argv) };
    }

    /// glibc passes argc, argv, and envp to functions in .init_array, as a non-standard extension.
    /// This allows `std::env::args` to work even in a `cdylib`, as it does on macOS and Windows.
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    #[used]
    #[unsafe(link_section = ".init_array.00099")]
    static ARGV_INIT_ARRAY: extern "C" fn(
        crate::os::raw::c_int,
        *const *const u8,
        *const *const u8,
    ) = {
        extern "C" fn init_wrapper(
            argc: crate::os::raw::c_int,
            argv: *const *const u8,
            _envp: *const *const u8,
        ) {
            unsafe { really_init(argc as isize, argv) };
        }
        init_wrapper
    };

    pub fn argc_argv() -> (isize, *const *const c_char) {
        // Load ARGC and ARGV, which hold the unmodified system-provided
        // argc/argv, so we can read the pointed-to memory without atomics or
        // synchronization.
        //
        // If either ARGC or ARGV is still zero or null, then either there
        // really are no arguments, or someone is asking for `args()` before
        // initialization has completed, and we return an empty list.
        let argv = ARGV.load(Ordering::Relaxed);
        let argc = if argv.is_null() { 0 } else { ARGC.load(Ordering::Relaxed) };

        // Cast from `*mut *const u8` to `*const *const c_char`
        (argc, argv.cast())
    }
}

// Use `_NSGetArgc` and `_NSGetArgv` on Apple platforms.
//
// Even though these have underscores in their names, they've been available
// since the first versions of both macOS and iOS, and are declared in
// the header `crt_externs.h`.
//
// NOTE: This header was added to the iOS 13.0 SDK, which has been the source
// of a great deal of confusion in the past about the availability of these
// APIs.
//
// NOTE(madsmtm): This has not strictly been verified to not cause App Store
// rejections; if this is found to be the case, the previous implementation
// of this used `[[NSProcessInfo processInfo] arguments]`.
#[cfg(target_vendor = "apple")]
mod imp {
    use crate::ffi::{c_char, c_int};

    pub unsafe fn init(_argc: isize, _argv: *const *const u8) {
        // No need to initialize anything in here, `libdyld.dylib` has already
        // done the work for us.
    }

    pub fn argc_argv() -> (isize, *const *const c_char) {
        unsafe extern "C" {
            // These functions are in crt_externs.h.
            fn _NSGetArgc() -> *mut c_int;
            fn _NSGetArgv() -> *mut *mut *mut c_char;
        }

        // SAFETY: The returned pointer points to a static initialized early
        // in the program lifetime by `libdyld.dylib`, and as such is always
        // valid.
        //
        // NOTE: Similar to `_NSGetEnviron`, there technically isn't anything
        // protecting us against concurrent modifications to this, and there
        // doesn't exist a lock that we can take. Instead, it is generally
        // expected that it's only modified in `main` / before other code
        // runs, so reading this here should be fine.
        let argc = unsafe { _NSGetArgc().read() };
        // SAFETY: Same as above.
        let argv = unsafe { _NSGetArgv().read() };

        // Cast from `*mut *mut c_char` to `*const *const c_char`
        (argc as isize, argv.cast())
    }
}
