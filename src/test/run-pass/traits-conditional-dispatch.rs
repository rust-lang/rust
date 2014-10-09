// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we are able to resolve conditional dispatch.  Here, the
// blanket impl for T:Copy coexists with an impl for Box<T>, because
// Box does not impl Copy.

trait Get {
    fn get(&self) -> Self;
}

impl<T:Copy> Get for T {
    fn get(&self) -> T { *self }
}

impl<T:Get> Get for Box<T> {
    fn get(&self) -> Box<T> { box get_it(&**self) }
}

fn get_it<T:Get>(t: &T) -> T {
    (*t).get()
}

fn main() {
    assert_eq!(get_it(&1_u32), 1_u32);
    assert_eq!(get_it(&1_u16), 1_u16);
    assert_eq!(get_it(&Some(1_u16)), Some(1_u16));
    assert_eq!(get_it(&box 1i), box 1i);
}
