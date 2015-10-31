// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Demonstrate that having a lifetime param causes dropck to reject code
// that might indirectly access previously dropped value.
//
// Compare with run-pass/issue28498-ugeh-with-lifetime-param.rs

#[derive(Debug)]
struct ScribbleOnDrop(String);

impl Drop for ScribbleOnDrop {
    fn drop(&mut self) {
        self.0 = format!("DROPPED");
    }
}

struct Foo<'a>(u32, &'a ScribbleOnDrop);

impl<'a> Drop for Foo<'a> {
    fn drop(&mut self) {
        // Use of `unsafe_destructor_blind_to_params` is unsound,
        // because destructor accesses borrowed data in `self.1`
        // and we must force that to strictly outlive `self`.
        println!("Dropping Foo({}, {:?})", self.0, self.1);
    }
}

fn main() {
    let (last_dropped, foo0);
    let (foo1, first_dropped);

    last_dropped = ScribbleOnDrop(format!("last"));
    first_dropped = ScribbleOnDrop(format!("first"));
    foo0 = Foo(0, &last_dropped);
    //~^ ERROR `last_dropped` does not live long enough
    foo1 = Foo(1, &first_dropped);
    //~^ ERROR `first_dropped` does not live long enough

    println!("foo0.1: {:?} foo1.1: {:?}", foo0.1, foo1.1);
}
