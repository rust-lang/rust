// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #22886.

fn crash_please() {
    let mut iter = Newtype(Some(Box::new(0)));
    let saved = iter.next().unwrap();
    println!("{}", saved);
    iter.0 = None;
    println!("{}", saved);
}

struct Newtype(Option<Box<usize>>);

impl<'a> Iterator for Newtype { //~ ERROR E0207
                                //~| NOTE unconstrained lifetime parameter
    type Item = &'a Box<usize>;

    fn next(&mut self) -> Option<&Box<usize>> {
        self.0.as_ref()
    }
}

fn main() { }
