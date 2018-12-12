// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[warn(clippy::precedence)]
#[allow(clippy::identity_op)]
#[allow(clippy::eq_op)]

macro_rules! trip {
    ($a:expr) => {
        match $a & 0b1111_1111i8 {
            0 => println!("a is zero ({})", $a),
            _ => println!("a is {}", $a),
        }
    };
}

fn main() {
    1 << 2 + 3;
    1 + 2 << 3;
    4 >> 1 + 1;
    1 + 3 >> 2;
    1 ^ 1 - 1;
    3 | 2 - 1;
    3 & 5 - 2;
    -1i32.abs();
    -1f32.abs();

    // These should not trigger an error
    let _ = (-1i32).abs();
    let _ = (-1f32).abs();
    let _ = -(1i32).abs();
    let _ = -(1f32).abs();
    let _ = -(1i32.abs());
    let _ = -(1f32.abs());

    let b = 3;
    trip!(b * 8);
}
