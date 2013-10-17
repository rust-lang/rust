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

use c_str::ToCStr;
use cast;
use libc::size_t;
use libc;
use repr;
use rt::task;
use str;

/// Returns the refcount of a shared box (as just before calling this)
#[inline]
pub fn refcount<T>(t: @T) -> uint {
    unsafe {
        let ref_ptr: *uint = cast::transmute_copy(&t);
        *ref_ptr - 1
    }
}

pub fn log_str<T>(t: &T) -> ~str {
    use rt::io;
    use rt::io::Decorator;

    let mut result = io::mem::MemWriter::new();
    repr::write_repr(&mut result as &mut io::Writer, t);
    str::from_utf8_owned(result.inner())
}

/// Trait for initiating task failure.
pub trait FailWithCause {
    /// Fail the current task, taking ownership of `cause`
    fn fail_with(cause: Self, file: &'static str, line: uint) -> !;
}

impl FailWithCause for ~str {
    fn fail_with(cause: ~str, file: &'static str, line: uint) -> ! {
        do cause.with_c_str |msg_buf| {
            do file.with_c_str |file_buf| {
                task::begin_unwind(msg_buf, file_buf, line as libc::size_t)
            }
        }
    }
}

impl FailWithCause for &'static str {
    fn fail_with(cause: &'static str, file: &'static str, line: uint) -> ! {
        do cause.with_c_str |msg_buf| {
            do file.with_c_str |file_buf| {
                task::begin_unwind(msg_buf, file_buf, line as libc::size_t)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use cast;
    use sys::*;

    #[test]
    fn synthesize_closure() {
        use unstable::raw::Closure;
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
