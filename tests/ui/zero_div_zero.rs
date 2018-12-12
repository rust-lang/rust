// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(unused_variables)]
#[warn(clippy::zero_divided_by_zero)]
fn main() {
    let nan = 0.0 / 0.0;
    let f64_nan = 0.0 / 0.0f64;
    let other_f64_nan = 0.0f64 / 0.0;
    let one_more_f64_nan = 0.0f64 / 0.0f64;
    let zero = 0.0;
    let other_zero = 0.0;
    let other_nan = zero / other_zero; // fine - this lint doesn't propegate constants.
    let not_nan = 2.0 / 0.0; // not an error: 2/0 = inf
    let also_not_nan = 0.0 / 2.0; // not an error: 0/2 = 0
}
