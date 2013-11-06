// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn main() {
    let a = 0xBEEF;
    let b = 0o755;
    let c = 0b10101;
    let d = -0xBEEF;
    let e = -0o755;
    let f = -0b10101;

    assert_eq!(a, 48879);
    assert_eq!(b, 493);
    assert_eq!(c, 21);
    assert_eq!(d, -48879);
    assert_eq!(e, -493);
    assert_eq!(f, -21);


}
