// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:trimmed-span.rs
// ignore-stage1

#![feature(proc_macro_non_items)]

extern crate trimmed_span;

use trimmed_span::trimmed;

fn main() {
    // This one emits no error.
    trimmed!("");

    // Exactly one 'hi'.
    trimmed!("hi"); //~ ERROR found 'hi's

    // Now two, back to back.
    trimmed!("hihi"); //~ ERROR found 'hi's

    // Now three, back to back.
    trimmed!("hihihi"); //~ ERROR found 'hi's

    // Now several, with spacing.
    trimmed!("why I hide? hi!"); //~ ERROR found 'hi's
    trimmed!("hey, hi, hidy, hidy, hi hi"); //~ ERROR found 'hi's
    trimmed!("this is a hi, and this is another hi"); //~ ERROR found 'hi's
    trimmed!("how are you this evening"); //~ ERROR found 'hi's
    trimmed!("this is highly eradic"); //~ ERROR found 'hi's
}
