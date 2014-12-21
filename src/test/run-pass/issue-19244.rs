// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct MyStruct { field: uint }
const STRUCT: MyStruct = MyStruct { field: 42 };
const TUP: (uint,) = (43,);

fn main() {
    let a = [0i; STRUCT.field];
    let b = [0i; TUP.0];

    assert!(a.len() == 42);
    assert!(b.len() == 43);
}
