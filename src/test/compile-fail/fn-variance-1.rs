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

fn takes_mut(x: @mut int) { }
fn takes_imm(x: @int) { }

fn apply<T>(t: T, f: |T|) {
    f(t)
}

fn main() {
    apply(@3, takes_mut); //~ ERROR (values differ in mutability)
    apply(@3, takes_imm);

    apply(@mut 3, takes_mut);
    apply(@mut 3, takes_imm); //~ ERROR (values differ in mutability)
}
