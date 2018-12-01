// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-fail

#![feature(never_type)]
#![feature(exhaustive_patterns)]

#![allow(unreachable_code)]
#![deny(unreachable_patterns)]

enum Void {}

impl Iterator for Void {
    type Item = Void;

    fn next(&mut self) -> Option<Void> {
        None
    }
}

fn main() {
    for _ in unimplemented!() as Void {}
    //~^ ERROR unreachable pattern
}

