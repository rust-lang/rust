// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME: this file probably shouldn't exist

#![macro_escape]

use std::fmt;

// Indicates whether we should perform expensive sanity checks, including rtassert!
// FIXME: Once the runtime matures remove the `true` below to turn off rtassert, etc.
pub static ENFORCE_SANITY: bool = true || !cfg!(rtopt) || cfg!(rtdebug) || cfg!(rtassert);

macro_rules! rterrln (
    ($($arg:tt)*) => ( {
        format_args!(::macros::dumb_println, $($arg)*)
    } )
)

// Some basic logging. Enabled by passing `--cfg rtdebug` to the libstd build.
macro_rules! rtdebug (
    ($($arg:tt)*) => ( {
        if cfg!(rtdebug) {
            rterrln!($($arg)*)
        }
    })
)

macro_rules! rtassert (
    ( $arg:expr ) => ( {
        if ::macros::ENFORCE_SANITY {
            if !$arg {
                rtabort!(" assertion failed: {}", stringify!($arg));
            }
        }
    } )
)


macro_rules! rtabort (
    ($($arg:tt)*) => ( {
        ::macros::abort(format!($($arg)*));
    } )
)

pub fn dumb_println(args: &fmt::Arguments) {
    use std::rt;
    let mut w = rt::Stderr;
    let _ = writeln!(&mut w, "{}", args);
}

pub fn abort(msg: &str) -> ! {
    let msg = if !msg.is_empty() { msg } else { "aborted" };
    rterrln!("fatal runtime error: {}", msg);

    abort();

    fn abort() -> ! {
        use std::intrinsics;
        unsafe { intrinsics::abort() }
    }
}
