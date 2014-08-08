// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![macro_escape]

macro_rules! uverrln (
    ($($arg:tt)*) => ( {
        use std::io::stdio;
        format_args!(stdio::println_args, $($arg)*)
    } )
)

// Some basic logging. Enabled by passing `--cfg uvdebug` to the libstd build.
macro_rules! uvdebug (
    ($($arg:tt)*) => ( {
        if cfg!(uvdebug) {
            uverrln!($($arg)*)
        }
    })
)

