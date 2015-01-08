// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test calling methods on an impl for a bare trait. This test checks that the
// trait impl is only applied to a trait object, not concrete types which implement
// the trait.

trait T {}

impl<'a> T+'a {
    fn foo(&self) {}
}

impl T for isize {}

fn main() {
    let x = &42is;
    x.foo(); //~ERROR: type `&isize` does not implement any method in scope named `foo`
}
