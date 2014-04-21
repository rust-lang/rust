// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn main() {
    let x = [1i, 2, 3];
    match x {
        [2, _, _] => fail!(),
        [1, a, b] => {
            assert!([a, b] == [2, 3]);
        }
        [_, _, _] => fail!(),
    }

    let y = ([(1i, true), (2i, false)], 0.5f64);
    match y {
        ([(1, a), (b, false)], _) => {
            assert_eq!(a, true);
            assert_eq!(b, 2);
        }
        ([_, _], 0.5) => fail!(),
        ([_, _], _) => fail!(),
    }
}
