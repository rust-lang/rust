// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that a class with an unsendable field can't be
// sent

use std::comm;

struct foo {
  i: int,
  j: @~str,
}

fn foo(i:int, j: @~str) -> foo {
    foo {
        i: i,
        j: j
    }
}

fn main() {
  let cat = ~"kitty";
    let (_, ch) = comm::stream(); //~ ERROR does not fulfill `Owned`
  ch.send(foo(42, @(cat))); //~ ERROR does not fulfill `Owned`
}
