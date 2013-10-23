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
}

fn borrowed_receiver(x: &Foo) {
    x.borrowed();
    x.borrowed_mut(); //~ ERROR cannot borrow
}

fn borrowed_mut_receiver(x: &mut Foo) {
    x.borrowed();
    x.borrowed_mut();
}

fn managed_receiver(x: @Foo) {
    x.borrowed();
    x.borrowed_mut(); //~ ERROR cannot borrow
}

fn managed_mut_receiver(x: @mut Foo) {
    x.borrowed();
    x.borrowed_mut();
}

fn owned_receiver(x: ~Foo) {
    x.borrowed();
    x.borrowed_mut(); //~ ERROR cannot borrow
}

fn mut_owned_receiver(mut x: ~Foo) {
    x.borrowed();
    x.borrowed_mut();
}

fn main() {}

