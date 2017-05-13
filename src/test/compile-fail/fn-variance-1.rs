// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn takes_imm(x: &isize) { }

fn takes_mut(x: &mut isize) { }

fn apply<T, F>(t: T, f: F) where F: FnOnce(T) {
    f(t)
}

fn main() {
    apply(&3, takes_imm);
    apply(&3, takes_mut);
    //~^ ERROR (types differ in mutability)

    apply(&mut 3, takes_mut);
    apply(&mut 3, takes_imm);
    //~^ ERROR (types differ in mutability)
}
