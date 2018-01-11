// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

union U1 { // OK
    a: u8,
}

union U2<T: Copy> { // OK
    a: T,
}

union U3 { //~ ERROR unions with non-`Copy` fields are unstable
    a: String,
}

union U4<T> { //~ ERROR unions with non-`Copy` fields are unstable
    a: T,
}

union U5 { //~ ERROR unions with `Drop` implementations are unstable
    a: u8,
}

impl Drop for U5 {
    fn drop(&mut self) {}
}

fn main() {}
