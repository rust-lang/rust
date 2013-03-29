// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[legacy_modes];

// In this test, the mode gets inferred to ++ due to the apply_int(),
// but then we get a failure in the generic apply().

fn apply<A>(f: &fn(A) -> A, a: A) -> A { f(a) }
fn apply_int(f: &fn(int) -> int, a: int) -> int { f(a) }

fn main() {
    let f = {|i| i};
    assert!(apply_int(f, 2) == 2);
    assert!(apply(f, 2) == 2); //~ ERROR expected argument mode &&
}
