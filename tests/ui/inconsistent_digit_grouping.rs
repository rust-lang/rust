// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[warn(clippy::inconsistent_digit_grouping)]
#[allow(unused_variables)]
fn main() {
    let good = (
        123,
        1_234,
        1_2345_6789,
        123_f32,
        1_234.12_f32,
        1_234.123_4_f32,
        1.123_456_7_f32,
    );
    let bad = (1_23_456, 1_234_5678, 1234_567, 1_234.5678_f32, 1.234_5678_f32);
}
