// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(specialization)]

// Test if two impls are allowed to overlap if a third
// impl is the intersection of them

trait MyClone {
    fn my_clone(&self) -> &'static str;
}

impl<T: Copy> MyClone for T {
    default fn my_clone(&self) -> &'static str {
        "impl_a"
    }
}

impl<T: Clone> MyClone for Option<T> {
    default fn my_clone(&self) -> &'static str {
        "impl_b"
    }
}

impl<T:Copy> MyClone for Option<T> {
    fn my_clone(&self) -> &'static str {
        "impl_c"
    }
}

fn main() {
    assert!(42i32.my_clone() == "impl_a");
    assert!(Some(Box::new(42i32)).my_clone() == "impl_b");
    assert!(Some(42i32).my_clone() == "impl_c");
}
