// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Misc low level stuff

#[allow(missing_doc)];

use option::{Some, None};
use cast;
use cmp::{Eq, Ord};
use gc;
use io;
use libc;
use libc::{c_void, c_char, size_t};
use repr;
use str;
use unstable::intrinsics;

pub type FreeGlue<'self> = &'self fn(*TypeDesc, *c_void);

// Corresponds to runtime type_desc type
pub struct TypeDesc {
    size: uint,
    align: uint,
    take_glue: uint,
    drop_glue: uint,
    free_glue: uint
    // Remaining fields not listed
}

/// The representation of a Rust closure
pub struct Closure {
    code: *(),
    env: *(),
}

pub mod rustrt {
    use libc::{c_char, size_t};

    pub extern {
        #[rust_stack]
        unsafe fn rust_upcall_fail(expr: *c_char,
                                   file: *c_char,
                                   line: size_t);
    }
}

/**
 * Returns a pointer to a type descriptor.
 *
 * Useful for calling certain function in the Rust runtime or otherwise
 * performing dark magick.
 */
#[inline(always)]
pub fn get_type_desc<T>() -> *TypeDesc {
    unsafe { intrinsics::get_tydesc::<T>() as *TypeDesc }
}

/// Returns a pointer to a type descriptor.
#[inline(always)]
pub fn get_type_desc_val<T>(_val: &T) -> *TypeDesc {
    get_type_desc::<T>()
}

/// Returns the size of a type
#[inline(always)]
pub fn size_of<T>() -> uint {
    unsafe { intrinsics::size_of::<T>() }
}

/// Returns the size of the type that `_val` points to
#[inline(always)]
pub fn size_of_val<T>(_val: &T) -> uint {
    size_of::<T>()
}

/**
 * Returns the size of a type, or 1 if the actual size is zero.
 *
 * Useful for building structures containing variable-length arrays.
 */
#[inline(always)]
pub fn nonzero_size_of<T>() -> uint {
    let s = size_of::<T>();
    if s == 0 { 1 } else { s }
}

/// Returns the size of the type of the value that `_val` points to
#[inline(always)]
pub fn nonzero_size_of_val<T>(_val: &T) -> uint {
    nonzero_size_of::<T>()
}


/**
 * Returns the ABI-required minimum alignment of a type
 *
 * This is the alignment used for struct fields. It may be smaller
 * than the preferred alignment.
 */
#[inline(always)]
pub fn min_align_of<T>() -> uint {
    unsafe { intrinsics::min_align_of::<T>() }
}

/// Returns the ABI-required minimum alignment of the type of the value that
/// `_val` points to
#[inline(always)]
pub fn min_align_of_val<T>(_val: &T) -> uint {
    min_align_of::<T>()
}

/// Returns the preferred alignment of a type
#[inline(always)]
pub fn pref_align_of<T>() -> uint {
    unsafe { intrinsics::pref_align_of::<T>() }
}

/// Returns the preferred alignment of the type of the value that
/// `_val` points to
#[inline(always)]
pub fn pref_align_of_val<T>(_val: &T) -> uint {
    pref_align_of::<T>()
}

/// Returns the refcount of a shared box (as just before calling this)
#[inline(always)]
pub fn refcount<T>(t: @T) -> uint {
    unsafe {
        let ref_ptr: *uint = cast::transmute_copy(&t);
        *ref_ptr - 1
    }
}

pub fn log_str<T>(t: &T) -> ~str {
    do io::with_str_writer |wr| {
        repr::write_repr(wr, t)
    }
}

/// Trait for initiating task failure.
pub trait FailWithCause {
    /// Fail the current task, taking ownership of `cause`
    fn fail_with(cause: Self, file: &'static str, line: uint) -> !;
}

impl FailWithCause for ~str {
    fn fail_with(cause: ~str, file: &'static str, line: uint) -> ! {
        do str::as_buf(cause) |msg_buf, _msg_len| {
            do str::as_buf(file) |file_buf, _file_len| {
                unsafe {
                    let msg_buf = cast::transmute(msg_buf);
                    let file_buf = cast::transmute(file_buf);
                    begin_unwind_(msg_buf, file_buf, line as libc::size_t)
                }
            }
        }
    }
}

impl FailWithCause for &'static str {
    fn fail_with(cause: &'static str, file: &'static str, line: uint) -> ! {
        do str::as_buf(cause) |msg_buf, _msg_len| {
            do str::as_buf(file) |file_buf, _file_len| {
                unsafe {
                    let msg_buf = cast::transmute(msg_buf);
                    let file_buf = cast::transmute(file_buf);
                    begin_unwind_(msg_buf, file_buf, line as libc::size_t)
                }
            }
        }
    }
}

// FIXME #4427: Temporary until rt::rt_fail_ goes away
pub fn begin_unwind_(msg: *c_char, file: *c_char, line: size_t) -> ! {
    use option::Option;
    use rt::{context, OldTaskContext, TaskContext};
    use rt::task::{Task, Unwinder};
    use rt::local::Local;

    let context = context();
    match context {
        OldTaskContext => {
            unsafe {
                gc::cleanup_stack_for_failure();
                rustrt::rust_upcall_fail(msg, file, line);
                cast::transmute(())
            }
        }
        _ => {
            unsafe {
                // XXX: Bad re-allocations. fail! needs some refactoring
                let msg = str::raw::from_c_str(msg);
                let file = str::raw::from_c_str(file);

                let outmsg = fmt!("%s at line %i of file %s", msg, line as int, file);

                // XXX: Logging doesn't work correctly in non-task context because it
                // invokes the local heap
                if context == TaskContext {
                    error!(outmsg);
                } else {
                    rtdebug!("%s", outmsg);
                }

                gc::cleanup_stack_for_failure();

                let task = Local::unsafe_borrow::<Task>();
                let unwinder: &mut Option<Unwinder> = &mut (*task).unwinder;
                match *unwinder {
                    Some(ref mut unwinder) => unwinder.begin_unwind(),
                    None => abort!("failure without unwinder. aborting process")
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use cast;
    use sys::*;

    #[test]
    fn size_of_basic() {
        assert_eq!(size_of::<u8>(), 1u);
        assert_eq!(size_of::<u16>(), 2u);
        assert_eq!(size_of::<u32>(), 4u);
        assert_eq!(size_of::<u64>(), 8u);
    }

    #[test]
    #[cfg(target_arch = "x86")]
    #[cfg(target_arch = "arm")]
    #[cfg(target_arch = "mips")]
    fn size_of_32() {
        assert_eq!(size_of::<uint>(), 4u);
        assert_eq!(size_of::<*uint>(), 4u);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn size_of_64() {
        assert_eq!(size_of::<uint>(), 8u);
        assert_eq!(size_of::<*uint>(), 8u);
    }

    #[test]
    fn size_of_val_basic() {
        assert_eq!(size_of_val(&1u8), 1);
        assert_eq!(size_of_val(&1u16), 2);
        assert_eq!(size_of_val(&1u32), 4);
        assert_eq!(size_of_val(&1u64), 8);
    }

    #[test]
    fn nonzero_size_of_basic() {
        type Z = [i8, ..0];
        assert_eq!(size_of::<Z>(), 0u);
        assert_eq!(nonzero_size_of::<Z>(), 1u);
        assert_eq!(nonzero_size_of::<uint>(), size_of::<uint>());
    }

    #[test]
    fn nonzero_size_of_val_basic() {
        let z = [0u8, ..0];
        assert_eq!(size_of_val(&z), 0u);
        assert_eq!(nonzero_size_of_val(&z), 1u);
        assert_eq!(nonzero_size_of_val(&1u), size_of_val(&1u));
    }

    #[test]
    fn align_of_basic() {
        assert_eq!(pref_align_of::<u8>(), 1u);
        assert_eq!(pref_align_of::<u16>(), 2u);
        assert_eq!(pref_align_of::<u32>(), 4u);
    }

    #[test]
    #[cfg(target_arch = "x86")]
    #[cfg(target_arch = "arm")]
    #[cfg(target_arch = "mips")]
    fn align_of_32() {
        assert_eq!(pref_align_of::<uint>(), 4u);
        assert_eq!(pref_align_of::<*uint>(), 4u);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn align_of_64() {
        assert_eq!(pref_align_of::<uint>(), 8u);
        assert_eq!(pref_align_of::<*uint>(), 8u);
    }

    #[test]
    fn align_of_val_basic() {
        assert_eq!(pref_align_of_val(&1u8), 1u);
        assert_eq!(pref_align_of_val(&1u16), 2u);
        assert_eq!(pref_align_of_val(&1u32), 4u);
    }

    #[test]
    fn synthesize_closure() {
        unsafe {
            let x = 10;
            let f: &fn(int) -> int = |y| x + y;

            assert_eq!(f(20), 30);

            let original_closure: Closure = cast::transmute(f);

            let actual_function_pointer = original_closure.code;
            let environment = original_closure.env;

            let new_closure = Closure {
                code: actual_function_pointer,
                env: environment
            };

            let new_f: &fn(int) -> int = cast::transmute(new_closure);
            assert_eq!(new_f(20), 30);
        }
    }

    #[test]
    #[should_fail]
    fn fail_static() { FailWithCause::fail_with("cause", file!(), line!())  }

    #[test]
    #[should_fail]
    fn fail_owned() { FailWithCause::fail_with(~"cause", file!(), line!())  }
}
