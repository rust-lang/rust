// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Z(u8, u8);

enum E {
    U(u8, u8),
}

fn main() {
    match Z(0, 1) {
        Z{..} => {} //~ ERROR tuple structs and variants in struct patterns are unstable
    }
    match E::U(0, 1) {
        E::U{..} => {} //~ ERROR tuple structs and variants in struct patterns are unstable
    }

    let z1 = Z(0, 1);
    let z2 = Z { ..z1 }; //~ ERROR tuple structs and variants in struct patterns are unstable
}
