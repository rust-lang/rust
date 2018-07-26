// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// gate-test-concat_idents

const XY_1: i32 = 10;

fn main() {
    const XY_2: i32 = 20;
    assert_eq!(10, concat_idents!(X, Y_1)); //~ ERROR `concat_idents` is not stable
    assert_eq!(20, concat_idents!(X, Y_2)); //~ ERROR `concat_idents` is not stable
}
