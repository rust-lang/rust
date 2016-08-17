// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// The `format_args!` syntax extension issues errors before code expansion
// has completed, but we still need a backtrace.

// This test includes stripped-down versions of `print!` and `println!`,
// because we can't otherwise verify the lines of the backtrace.

fn print(_args: std::fmt::Arguments) {}

macro_rules! myprint {
    ($($arg:tt)*) => (print(format_args!($($arg)*)));
}

macro_rules! myprintln {
    ($fmt:expr) => (myprint!(concat!($fmt, "\n")));
}

fn main() {
    myprintln!("{}");
}
