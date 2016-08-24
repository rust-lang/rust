// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn const_mir() -> f32 { 9007199791611905.0 }

fn main() {
    let original = "9007199791611905.0"; // (1<<53)+(1<<29)+1
    let expected = "9007200000000000";

    assert_eq!(const_mir().to_string(), expected);
    assert_eq!(original.parse::<f32>().unwrap().to_string(), expected);
}
