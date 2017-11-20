// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let a = [(1u32, 2u32)];
    a.iter().map(|_: (u32, u32)| 45); //~ ERROR type mismatch
    a.iter().map(|_: &(u16, u16)| 45); //~ ERROR type mismatch
    a.iter().map(|_: (u16, u16)| 45); //~ ERROR type mismatch
}

fn baz<F: Fn(*mut &u32)>(_: F) {}
fn _test<'a>(f: fn(*mut &'a u32)) {
    baz(f); //~ ERROR type mismatch
     //~^ ERROR type mismatch
}
