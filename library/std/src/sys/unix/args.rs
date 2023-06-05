//! Global initialization and retrieval of command line arguments.
//!
//! On some platforms these are stored during runtime startup,
//! and on some they are retrieved from the system on demand.

#![allow(dead_code)] // runtime init functions not used during testing

use crate::ffi::OsString;
use crate::fmt;
use crate::vec;

/// One-time global initialization.
pub unsafe fn init(argc: isize, argv: *const *const u8) {
    imp::init(argc, argv)
}

/// Returns the command line arguments
pub fn args() -> Args {
    imp::args()
}

pub struct Args {
    iter: vec::IntoIter<OsString>,
}

impl !Send for Args {}
impl !Sync for Args {}

impl fmt::Debug for Args {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.iter.as_slice().fmt(f)
    }
}

impl Iterator for Args {
    type Item = OsString;
    fn next(&mut self) -> Option<OsString> {
        self.iter.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl ExactSizeIterator for Args {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl DoubleEndedIterator for Args {
    fn next_back(&mut self) -> Option<OsString> {
        self.iter.next_back()
    }
}

#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "freebsd",
    target_os = "dragonfly",
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "solaris",
    target_os = "illumos",
    target_os = "emscripten",
    target_os = "haiku",
    target_os = "l4re",
    target_os = "fuchsia",
    target_os = "redox",
    target_os = "vxworks",
    target_os = "horizon",
    target_os = "nto",
))]
mod imp {
    use super::Args;
    use crate::ffi::{CStr, OsString};
    use crate::os::unix::prelude::*;
    use crate::ptr;
    use crate::sync::atomic::{AtomicIsize, AtomicPtr, Ordering};

    // The system-provided argc and argv, which we store in static memory
    // here so that we can defer the work of parsing them until its actually
    // needed.
    //
    // Note that we never mutate argv/argc, the argv array, or the argv
    // strings, which allows the code in this file to be very simple.
    static ARGC: AtomicIsize = AtomicIsize::new(0);
    static ARGV: AtomicPtr<*const u8> = AtomicPtr::new(ptr::null_mut());

    unsafe fn really_init(argc: isize, argv: *const *const u8) {
        // These don't need to be ordered with each other or other stores,
        // because they only hold the unmodified system-provide argv/argc.
        ARGC.store(argc, Ordering::Relaxed);
        ARGV.store(argv as *mut _, Ordering::Relaxed);
    }

    #[inline(always)]
    pub unsafe fn init(_argc: isize, _argv: *const *const u8) {
        // On Linux-GNU, we rely on `ARGV_INIT_ARRAY` below to initialize
        // `ARGC` and `ARGV`. But in Miri that does not actually happen so we
        // still initialize here.
        #[cfg(any(miri, not(all(target_os = "linux", target_env = "gnu"))))]
        really_init(_argc, _argv);
    }

    /// glibc passes argc, argv, and envp to functions in .init_array, as a non-standard extension.
    /// This allows `std::env::args` to work even in a `cdylib`, as it does on macOS and Windows.
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    #[used]
    #[link_section = ".init_array.00099"]
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
            unsafe {
                really_init(argc as isize, argv);
            }
        }
        init_wrapper
    };

    pub fn args() -> Args {
        Args { iter: clone().into_iter() }
    }

    fn clone() -> Vec<OsString> {
        unsafe {
            // Load ARGC and ARGV, which hold the unmodified system-provided
            // argc/argv, so we can read the pointed-to memory without atomics
            // or synchronization.
            //
            // If either ARGC or ARGV is still zero or null, then either there
            // really are no arguments, or someone is asking for `args()`
            // before initialization has completed, and we return an empty
            // list.
            let argv = ARGV.load(Ordering::Relaxed);
            let argc = if argv.is_null() { 0 } else { ARGC.load(Ordering::Relaxed) };
            let mut args = Vec::with_capacity(argc as usize);
            for i in 0..argc {
                let ptr = *argv.offset(i) as *const libc::c_char;

                // Some C commandline parsers (e.g. GLib and Qt) are replacing already
                // handled arguments in `argv` with `NULL` and move them to the end. That
                // means that `argc` might be bigger than the actual number of non-`NULL`
                // pointers in `argv` at this point.
                //
                // To handle this we simply stop iterating at the first `NULL` argument.
                //
                // `argv` is also guaranteed to be `NULL`-terminated so any non-`NULL` arguments
                // after the first `NULL` can safely be ignored.
                if ptr.is_null() {
                    break;
                }

                let cstr = CStr::from_ptr(ptr);
                args.push(OsStringExt::from_vec(cstr.to_bytes().to_vec()));
            }

            args
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "ios", target_os = "watchos"))]
mod imp {
    use super::Args;
    use crate::ffi::CStr;

    pub unsafe fn init(_argc: isize, _argv: *const *const u8) {}

    #[cfg(target_os = "macos")]
    pub fn args() -> Args {
        use crate::os::unix::prelude::*;
        extern "C" {
            // These functions are in crt_externs.h.
            fn _NSGetArgc() -> *mut libc::c_int;
            fn _NSGetArgv() -> *mut *mut *mut libc::c_char;
        }

        let vec = unsafe {
            let (argc, argv) =
                (*_NSGetArgc() as isize, *_NSGetArgv() as *const *const libc::c_char);
            (0..argc as isize)
                .map(|i| {
                    let bytes = CStr::from_ptr(*argv.offset(i)).to_bytes().to_vec();
                    OsStringExt::from_vec(bytes)
                })
                .collect::<Vec<_>>()
        };
        Args { iter: vec.into_iter() }
    }

    // As _NSGetArgc and _NSGetArgv aren't mentioned in iOS docs
    // and use underscores in their names - they're most probably
    // are considered private and therefore should be avoided
    // Here is another way to get arguments using Objective C
    // runtime
    //
    // In general it looks like:
    // res = Vec::new()
    // let args = [[NSProcessInfo processInfo] arguments]
    // for i in (0..[args count])
    //      res.push([args objectAtIndex:i])
    // res
    #[cfg(any(target_os = "ios", target_os = "watchos"))]
    pub fn args() -> Args {
        use crate::ffi::OsString;
        use crate::mem;
        use crate::str;

        extern "C" {
            fn sel_registerName(name: *const libc::c_uchar) -> Sel;
            fn objc_getClass(class_name: *const libc::c_uchar) -> NsId;
        }

        #[cfg(target_arch = "aarch64")]
        extern "C" {
            fn objc_msgSend(obj: NsId, sel: Sel) -> NsId;
            #[allow(clashing_extern_declarations)]
            #[link_name = "objc_msgSend"]
            fn objc_msgSend_ul(obj: NsId, sel: Sel, i: libc::c_ulong) -> NsId;
        }

        #[cfg(not(target_arch = "aarch64"))]
        extern "C" {
            fn objc_msgSend(obj: NsId, sel: Sel, ...) -> NsId;
            #[allow(clashing_extern_declarations)]
            #[link_name = "objc_msgSend"]
            fn objc_msgSend_ul(obj: NsId, sel: Sel, ...) -> NsId;
        }

        type Sel = *const libc::c_void;
        type NsId = *const libc::c_void;

        let mut res = Vec::new();

        unsafe {
            let process_info_sel =
                sel_registerName(c"processInfo".as_ptr() as *const libc::c_uchar);
            let arguments_sel = sel_registerName(c"arguments".as_ptr() as *const libc::c_uchar);
            let utf8_sel = sel_registerName(c"UTF8String".as_ptr() as *const libc::c_uchar);
            let count_sel = sel_registerName(c"count".as_ptr() as *const libc::c_uchar);
            let object_at_sel =
                sel_registerName(c"objectAtIndex:".as_ptr() as *const libc::c_uchar);

            let klass = objc_getClass(c"NSProcessInfo".as_ptr() as *const libc::c_uchar);
            let info = objc_msgSend(klass, process_info_sel);
            let args = objc_msgSend(info, arguments_sel);

            let cnt: usize = mem::transmute(objc_msgSend(args, count_sel));
            for i in 0..cnt {
                let tmp = objc_msgSend_ul(args, object_at_sel, i as libc::c_ulong);
                let utf_c_str: *const libc::c_char = mem::transmute(objc_msgSend(tmp, utf8_sel));
                let bytes = CStr::from_ptr(utf_c_str).to_bytes();
                res.push(OsString::from(str::from_utf8(bytes).unwrap()))
            }
        }

        Args { iter: res.into_iter() }
    }
}

#[cfg(any(target_os = "espidf", target_os = "vita"))]
mod imp {
    use super::Args;

    #[inline(always)]
    pub unsafe fn init(_argc: isize, _argv: *const *const u8) {}

    pub fn args() -> Args {
        Args { iter: Vec::new().into_iter() }
    }
}
