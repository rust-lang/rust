// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Global storage for command line arguments
//!
//! The current incarnation of the Rust runtime expects for
//! the processes `argc` and `argv` arguments to be stored
//! in a globally-accessible location for use by the `os` module.
//!
//! Only valid to call on linux. Mac and Windows use syscalls to
//! discover the command line arguments.
//!
//! FIXME #7756: Would be nice for this to not exist.
//! FIXME #7756: This has a lot of C glue for lack of globals.

use option::Option;

/// One-time global initialization.
pub unsafe fn init(argc: int, argv: **u8) {
    imp::init(argc, argv)
}

/// One-time global cleanup.
pub fn cleanup() {
    imp::cleanup()
}

/// Take the global arguments from global storage.
pub fn take() -> Option<~[~str]> {
    imp::take()
}

/// Give the global arguments to global storage.
///
/// It is an error if the arguments already exist.
pub fn put(args: ~[~str]) {
    imp::put(args)
}

/// Make a clone of the global arguments.
pub fn clone() -> Option<~[~str]> {
    imp::clone()
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "android")]
#[cfg(target_os = "freebsd")]
mod imp {
    use libc;
    use option::{Option, Some, None};
    use iter::Iterator;
    use str;
    use unstable::finally::Finally;
    use util;
    use vec;

    pub unsafe fn init(argc: int, argv: **u8) {
        let args = load_argc_and_argv(argc, argv);
        put(args);
    }

    pub fn cleanup() {
        rtassert!(take().is_some());
    }

    pub fn take() -> Option<~[~str]> {
        with_lock(|| unsafe {
            let ptr = get_global_ptr();
            let val = util::replace(&mut *ptr, None);
            val.as_ref().map(|s: &~~[~str]| (**s).clone())
        })
    }

    pub fn put(args: ~[~str]) {
        with_lock(|| unsafe {
            let ptr = get_global_ptr();
            rtassert!((*ptr).is_none());
            (*ptr) = Some(~args.clone());
        })
    }

    pub fn clone() -> Option<~[~str]> {
        with_lock(|| unsafe {
            let ptr = get_global_ptr();
            (*ptr).as_ref().map(|s: &~~[~str]| (**s).clone())
        })
    }

    fn with_lock<T>(f: &fn() -> T) -> T {
        do (|| {
            unsafe {
                rust_take_global_args_lock();
                f()
            }
        }).finally {
            unsafe {
                rust_drop_global_args_lock();
            }
        }
    }

    fn get_global_ptr() -> *mut Option<~~[~str]> {
        unsafe { rust_get_global_args_ptr() }
    }

    // Copied from `os`.
    unsafe fn load_argc_and_argv(argc: int, argv: **u8) -> ~[~str] {
        do vec::from_fn(argc as uint) |i| {
            str::raw::from_c_str(*(argv as **libc::c_char).offset(i as int))
        }
    }

    externfn!(fn rust_take_global_args_lock())
    externfn!(fn rust_drop_global_args_lock())
    externfn!(fn rust_get_global_args_ptr() -> *mut Option<~~[~str]>)

    #[cfg(test)]
    mod tests {
        use option::{Some, None};
        use super::*;
        use unstable::finally::Finally;

        #[test]
        fn smoke_test() {
            // Preserve the actual global state.
            let saved_value = take();

            let expected = ~[~"happy", ~"today?"];

            put(expected.clone());
            assert!(clone() == Some(expected.clone()));
            assert!(take() == Some(expected.clone()));
            assert!(take() == None);

            do (|| {
            }).finally {
                // Restore the actual global state.
                match saved_value {
                    Some(ref args) => put(args.clone()),
                    None => ()
                }
            }
        }
    }
}

#[cfg(target_os = "macos")]
#[cfg(target_os = "win32")]
mod imp {
    use option::Option;

    pub unsafe fn init(_argc: int, _argv: **u8) {
    }

    pub fn cleanup() {
    }

    pub fn take() -> Option<~[~str]> {
        fail2!()
    }

    pub fn put(_args: ~[~str]) {
        fail2!()
    }

    pub fn clone() -> Option<~[~str]> {
        fail2!()
    }
}
