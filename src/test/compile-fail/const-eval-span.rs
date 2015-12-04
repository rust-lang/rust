// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that error in constant evaluation of enum discriminant
// provides the context for what caused the evaluation.

struct S(i32);

const CONSTANT: S = S(0);
//~^ ERROR: constant evaluation error: call on struct [E0080]

enum E {
    V = CONSTANT,
    //~^ NOTE: for enum discriminant here
}

fn main() {}
