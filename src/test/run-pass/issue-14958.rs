// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unboxed_closures)]

trait Foo { fn dummy(&self) { }}

struct Bar;

impl<'a> std::ops::Fn<(&'a (Foo+'a),)> for Bar {
    type Output = ();
    extern "rust-call" fn call(&self, _: (&'a Foo,)) {}
}

struct Baz;

impl Foo for Baz {}

fn main() {
    let bar = Bar;
    let baz = &Baz;
    bar(baz);
}
