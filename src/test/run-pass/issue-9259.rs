// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct A<'a> {
    a: &'a [String],
    b: Option<&'a [String]>,
}

pub fn main() {
    let b: &[String] = &["foo".to_string()];
    let a = A {
        a: &["test".to_string()],
        b: Some(b),
    };
    assert_eq!(a.b.get_ref()[0].as_slice(), "foo");
}
