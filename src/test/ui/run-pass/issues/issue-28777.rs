// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let v1 = { 1 + {2} * {3} };
    let v2 =   1 + {2} * {3}  ;

    assert_eq!(7, v1);
    assert_eq!(7, v2);

    let v3;
    v3 = { 1 + {2} * {3} };
    let v4;
    v4 = 1 + {2} * {3};
    assert_eq!(7, v3);
    assert_eq!(7, v4);

    let v5 = { 1 + {2} * 3 };
    assert_eq!(7, v5);

    let v9 = { 1 + if 1 > 2 {1} else {2} * {3} };
    assert_eq!(7, v9);
}
