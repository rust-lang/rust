// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::sys::size_of;

#[no_drop_flag]
struct Test<T> {
    a: T
}

#[unsafe_destructor]
impl<T> Drop for Test<T> {
    fn drop(&self) { }
}

fn main() {
    assert_eq!(size_of::<int>(), size_of::<Test<int>>());
}
