// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:unwind happens
// error-pattern:drop 3
// error-pattern:drop 2
// error-pattern:drop 1
use std::io::{self, Write};


/// Structure which will not allow to be dropped twice.
struct Droppable<'a>(&'a mut bool, u32);
impl<'a> Drop for Droppable<'a> {
    fn drop(&mut self) {
        if *self.0 {
            writeln!(io::stderr(), "{} dropped twice", self.1);
            ::std::process::exit(1);
        }
        writeln!(io::stderr(), "drop {}", self.1);
        *self.0 = true;
    }
}

fn may_panic<'a>() -> Droppable<'a> {
    panic!("unwind happens");
}

fn mir<'a>(d: Droppable<'a>) {
    let (mut a, mut b) = (false, false);
    let y = Droppable(&mut a, 2);
    let x = [Droppable(&mut b, 1), y, d, may_panic()];
}

fn main() {
    let mut c = false;
    mir(Droppable(&mut c, 3));
}
