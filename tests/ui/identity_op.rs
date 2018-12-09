// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

const ONE: i64 = 1;
const NEG_ONE: i64 = -1;
const ZERO: i64 = 0;

#[allow(
    clippy::eq_op,
    clippy::no_effect,
    clippy::unnecessary_operation,
    clippy::double_parens
)]
#[warn(clippy::identity_op)]
fn main() {
    let x = 0;

    x + 0;
    x + (1 - 1);
    x + 1;
    0 + x;
    1 + x;
    x - ZERO; //no error, as we skip lookups (for now)
    x | (0);
    (ZERO) | x; //no error, as we skip lookups (for now)

    x * 1;
    1 * x;
    x / ONE; //no error, as we skip lookups (for now)

    x / 2; //no false positive

    x & NEG_ONE; //no error, as we skip lookups (for now)
    -1 & x;

    let u: u8 = 0;
    u & 255;
}
