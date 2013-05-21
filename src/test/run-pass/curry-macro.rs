// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

curry!(
    fn mul(x: int, y: int, z: int) -> int {
        x * y * z
    }
)

fn main() {
    assert_eq!(mul(2)(3)(2), 12);

    let mul2 = mul(2);
    assert_eq!(mul2(3)(2), 12);

    // type error :(
    // assert_eq!(curry!(|x,y,z| x * y * z)(2)(3)(2), 12);

    let local_mul: @fn(int) -> @fn(int) -> @fn(int) -> int = curry!(|x,y,z| x * y * z);
    assert_eq!(local_mul(2)(3)(2), 12);
}
