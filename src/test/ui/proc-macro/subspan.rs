// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:subspan.rs

extern crate subspan;

use subspan::subspan;

// This one emits no error.
subspan!("");

// Exactly one 'hi'.
subspan!("hi"); //~ ERROR found 'hi's

// Now two, back to back.
subspan!("hihi"); //~ ERROR found 'hi's

// Now three, back to back.
subspan!("hihihi"); //~ ERROR found 'hi's

// Now several, with spacing.
subspan!("why I hide? hi!"); //~ ERROR found 'hi's
subspan!("hey, hi, hidy, hidy, hi hi"); //~ ERROR found 'hi's
subspan!("this is a hi, and this is another hi"); //~ ERROR found 'hi's
subspan!("how are you this evening"); //~ ERROR found 'hi's
subspan!("this is highly eradic"); //~ ERROR found 'hi's

fn main() { }
