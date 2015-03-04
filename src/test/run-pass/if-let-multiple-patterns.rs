// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum Foo { One(u8), Two(u8), Three }
use Foo::*;

fn main () {
    let x = Two(42);
    let mut result = 0;

    if let One(n) | Two(n) = x  {
        result = n;
    }


    assert_eq!(result, 42)
}
