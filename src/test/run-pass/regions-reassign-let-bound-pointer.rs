// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that the type checker permits us to reassign `z` which
// started out with a longer lifetime and was reassigned to a shorter
// one (it should infer to be the intersection).

fn foo(x: &int) {
    let a = 1;
    let mut z = x;
    z = &a;
}

pub fn main() {
    foo(&1);
}
