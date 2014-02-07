// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that an `&` pointer to something inherently mutable is itself
// to be considered mutable.

use std::kinds::marker;

enum Foo { A(marker::NoFreeze) }

fn bar<T: Freeze>(_: T) {}

fn main() {
    let x = A(marker::NoFreeze);
    bar(&x); //~ ERROR type parameter with an incompatible type
}
