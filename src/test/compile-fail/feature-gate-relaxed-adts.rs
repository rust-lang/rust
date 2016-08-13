// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct S(); //~ ERROR empty tuple structs and enum variants are unstable
struct Z(u8, u8);

enum E {
    V(), //~ ERROR empty tuple structs and enum variants are unstable
    U(u8, u8),
}

fn main() {
    match S() {
        S() => {} //~ ERROR empty tuple structs patterns are unstable
    }
    match E::V() {
        E::V() => {} //~ ERROR empty tuple structs patterns are unstable
    }
}
