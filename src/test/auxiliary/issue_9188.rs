// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn foo<T>() -> &'static isize {
    if false {
        static a: isize = 4;
        return &a;
    } else {
        static a: isize = 5;
        return &a;
    }
}

pub fn bar() -> &'static isize {
    foo::<isize>()
}
