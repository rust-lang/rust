// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn tuple() {
    let x = (1, 2, 3);
    let branch = match x {
        (1, 1, ..) => 0,
        (1, 2, 3, ..) => 1,
        (1, 2, ..) => 2,
        _ => 3
    };
    assert_eq!(branch, 1);
}

fn tuple_struct() {
    struct S(u8, u8, u8);

    let x = S(1, 2, 3);
    let branch = match x {
        S(1, 1, ..) => 0,
        S(1, 2, 3, ..) => 1,
        S(1, 2, ..) => 2,
        _ => 3
    };
    assert_eq!(branch, 1);
}

fn main() {
    tuple();
    tuple_struct();
}
