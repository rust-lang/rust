// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

#![feature(const_indexing)]

const FOO: [u32; 3] = [1, 2, 3];
const BAR: u32 = FOO[5]; // no error, because the error below occurs before regular const eval

const BLUB: [u32; FOO[4]] = [5, 6];
//~^ ERROR constant evaluation error [E0080]
//~| index out of bounds: the len is 3 but the index is 4

fn main() {
    let _ = BAR;
}
