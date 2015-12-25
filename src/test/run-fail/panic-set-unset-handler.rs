// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:thread '<main>' panicked at 'foobar'

#![feature(std_panic, panic_handler)]
use std::panic;
use std::io::{self, Write};

fn main() {
    panic::set_handler(|i| {
        write!(io::stderr(), "greetings from the panic handler");
    });
    panic::take_handler();
    panic!("foobar");
}
