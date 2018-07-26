// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

macro_rules! foo {
    ($l:lifetime, $l2:lifetime) => {
        fn f<$l: $l2, $l2>(arg: &$l str, arg2: &$l2 str) -> &$l str {
            arg
        }
    }
}

pub fn main() {
    foo!('a, 'b);
    let x: &'static str = f("hi", "there");
    assert_eq!("hi", x);
}
