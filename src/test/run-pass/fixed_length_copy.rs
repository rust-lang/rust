// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// error on implicit copies to check fixed length vectors
// are implicitly copyable
#[deny(implicit_copies)]
pub fn main() {
    let arr = [1,2,3];
    let arr2 = arr;
    assert_eq!(arr[1], 2);
    assert_eq!(arr2[2], 3);
}
