// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]
#![warn(clippy::multiple_inherent_impl)]

struct MyStruct;

impl MyStruct {
    fn first() {}
}

impl MyStruct {
    fn second() {}
}

impl<'a> MyStruct {
    fn lifetimed() {}
}

mod submod {
    struct MyStruct;
    impl MyStruct {
        fn other() {}
    }

    impl super::MyStruct {
        fn third() {}
    }
}

use std::fmt;
impl fmt::Debug for MyStruct {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "MyStruct {{ }}")
    }
}

fn main() {}
