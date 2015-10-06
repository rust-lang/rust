// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Example taken from RFC 1238 text

// https://github.com/rust-lang/rfcs/blob/master/text/1238-nonparametric-dropck.md#examples-of-code-that-will-start-to-be-rejected

// Compare against test/run-pass/issue28498-must-work-ex2.rs

use std::cell::Cell;

#[derive(Copy, Clone, Debug)]
enum Validity { Valid, Invalid }
use self::Validity::{Valid, Invalid};

struct Abstract<T> {
    id: u32,
    nbor: Cell<Option<T>>,
    valid: Validity,
    observe: fn(&Cell<Option<T>>) -> (u32, Validity),
}

#[derive(Copy, Clone)]
struct Neighbor<'a>(&'a Abstract<Neighbor<'a>>);

fn observe(c: &Cell<Option<Neighbor>>) -> (u32, Validity) {
    let r = c.get().unwrap().0;
    (r.id, r.valid)
}

impl<'a> Abstract<Neighbor<'a>> {
    fn new(id: u32) -> Self {
        Abstract {
            id: id,
            nbor: Cell::new(None),
            valid: Valid,
            observe: observe
        }
    }
}

struct Foo<T> {
    data: Vec<T>,
}

impl<T> Drop for Abstract<T> {
    fn drop(&mut self) {
        let (nbor_id, nbor_valid) = (self.observe)(&self.nbor);
        println!("dropping element {} ({:?}), observed neighbor {} ({:?})",
                 self.id,
                 self.valid,
                 nbor_id,
                 nbor_valid);
        self.valid = Invalid;
    }
}

fn main() {
    let mut foo: Foo<Abstract<Neighbor>> = Foo {  data: Vec::new() };
    foo.data.push(Abstract::new(0));
    foo.data.push(Abstract::new(1));

    foo.data[0].nbor.set(Some(Neighbor(&foo.data[1])));
    //~^ ERROR `foo.data` does not live long enough
    foo.data[1].nbor.set(Some(Neighbor(&foo.data[0])));
    //~^ ERROR `foo.data` does not live long enough
}
