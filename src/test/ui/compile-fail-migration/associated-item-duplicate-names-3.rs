// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// Before the introduction of the "duplicate associated type" error, the
// program below used to result in the "ambiguous associated type" error E0223,
// which is unexpected.

trait Foo {
    type Bar;
}

struct Baz;

impl Foo for Baz {
    type Bar = i16;
    type Bar = u16; //~ ERROR duplicate definitions
}

fn main() {
    let x: Baz::Bar = 5;
}
