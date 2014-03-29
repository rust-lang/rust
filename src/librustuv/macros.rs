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

use std::fmt;

macro_rules! uverrln (
    ($($arg:tt)*) => ( {
        format_args!(::macros::dumb_println, $($arg)*)
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

pub fn dumb_println(args: &fmt::Arguments) {
    use std::io;
    use std::libc;

    struct Stderr;
    impl io::Writer for Stderr {
        fn write(&mut self, data: &[u8]) -> io::IoResult<()> {
            let _ = unsafe {
                libc::write(libc::STDERR_FILENO,
                            data.as_ptr() as *libc::c_void,
                            data.len() as libc::size_t)
            };
            Ok(()) // just ignore the errors
        }
    }
    let mut w = Stderr;
    let _ = fmt::writeln(&mut w as &mut io::Writer, args);
}
