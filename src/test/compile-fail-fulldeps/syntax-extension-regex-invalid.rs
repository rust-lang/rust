// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME(#13725) windows needs fixing.
// ignore-win32
// ignore-stage1

#![feature(phase)]

extern crate regex;
#[phase(plugin)] extern crate regex_macros;

// Tests to make sure that `regex!` will produce a compile error when given
// an invalid regular expression.
// More exhaustive failure tests for the parser are done with the traditional
// unit testing infrastructure, since both dynamic and native regexes use the
// same parser.

fn main() {
    let _ = regex!("("); //~ ERROR Regex syntax error
}
