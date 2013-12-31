// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(managed_boxes)];

trait Foo {
    fn borrowed(&self);
    fn borrowed_mut(&mut self);

    fn managed(@self);

    fn owned(~self);
}

fn borrowed_receiver(x: &Foo) {
    x.borrowed();
    x.borrowed_mut(); // See [1]
    x.managed(); //~ ERROR does not implement any method
    x.owned(); //~ ERROR does not implement any method
}

fn borrowed_mut_receiver(x: &mut Foo) {
    x.borrowed();
    x.borrowed_mut();
    x.managed(); //~ ERROR does not implement any method
    x.owned(); //~ ERROR does not implement any method
}

fn managed_receiver(x: @Foo) {
    x.borrowed();
    x.borrowed_mut(); // See [1]
    x.managed();
    x.owned(); //~ ERROR does not implement any method
}

fn owned_receiver(x: ~Foo) {
    x.borrowed();
    x.borrowed_mut(); // See [1]
    x.managed();  //~ ERROR does not implement any method
    x.owned();
}

fn main() {}

// [1]: These cases are illegal, but the error is not detected
// until borrowck, so see the test borrowck-object-mutability.rs
