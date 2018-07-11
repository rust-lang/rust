// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(nll)]

struct X(usize);

impl X {
    fn zap(&self) {
        //~^ HELP
        //~| SUGGESTION &mut self
        self.0 = 32;
        //~^ ERROR
    }
}

fn main() {
    let ref foo = 16;
    //~^ HELP
    //~| SUGGESTION ref mut foo
    *foo = 32;
    //~^ ERROR
    if let Some(ref bar) = Some(16) {
        //~^ HELP
        //~| SUGGESTION ref mut bar
        *bar = 32;
        //~^ ERROR
    }
    match 16 {
        ref quo => { *quo = 32; },
        //~^ ERROR
        //~| HELP
        //~| SUGGESTION ref mut quo
    }
}
