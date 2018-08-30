// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// #30527 - We were not generating arms with guards in certain cases.

fn match_with_guard(x: Option<i8>) -> i8 {
    match x {
        Some(xyz) if xyz > 100 => 0,
        Some(_) => -1,
        None => -2
    }
}

fn main() {
    assert_eq!(match_with_guard(Some(111)), 0);
    assert_eq!(match_with_guard(Some(2)), -1);
    assert_eq!(match_with_guard(None), -2);
}
