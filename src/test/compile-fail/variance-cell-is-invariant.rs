// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that Cell is considered invariant with respect to its
// type.

use std::cell::Cell;

struct Foo<'a> {
    x: Cell<Option<&'a int>>,
}

fn use_<'short,'long>(c: Foo<'short>,
                      s: &'short int,
                      l: &'long int,
                      _where:Option<&'short &'long ()>) {
    let _: Foo<'long> = c; //~ ERROR mismatched types
}

fn main() {
}
