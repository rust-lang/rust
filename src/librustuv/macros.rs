// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[macro_escape];

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

// get a handle for the current scheduler
macro_rules! get_handle_to_current_scheduler(
    () => (do Local::borrow |sched: &mut Scheduler| { sched.make_handle() })
)

pub fn dumb_println(args: &fmt::Arguments) {
    use std::io::native::file::FileDesc;
    use std::io;
    use std::libc;
    let mut out = FileDesc::new(libc::STDERR_FILENO, false);
    fmt::writeln(&mut out as &mut io::Writer, args);
}
