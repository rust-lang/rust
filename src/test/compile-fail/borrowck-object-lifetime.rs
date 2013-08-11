// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Foo {
    fn borrowed<'a>(&'a self) -> &'a ();
}

fn borrowed_receiver<'a>(x: &'a Foo) -> &'a () {
    x.borrowed()
}

fn managed_receiver(x: @Foo) -> &() {
    x.borrowed() //~ ERROR cannot root managed value long enough
}

fn managed_receiver_1(x: @Foo) {
    *x.borrowed()
}

fn owned_receiver(x: ~Foo) -> &() {
    x.borrowed() //~ ERROR borrowed value does not live long enough
}

fn mut_owned_receiver(mut x: ~Foo) {
    let _y = x.borrowed();
    let _z = &mut x; //~ ERROR cannot borrow
}

fn imm_owned_receiver(mut x: ~Foo) {
    let _y = x.borrowed();
    let _z = &x;
}

fn main() {}

