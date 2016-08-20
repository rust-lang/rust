// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that an unsafe impl does not imply that unsafe actions are
// legal in the methods.

unsafe trait UnsafeTrait : Sized {
    fn foo(self) { }
}

unsafe impl UnsafeTrait for *mut isize {
    fn foo(self) {
        // Unsafe actions are not made legal by taking place in an unsafe trait:
        *self += 1;
        //~^ ERROR E0133
        //~| NOTE dereference of raw pointer
    }
}

fn main() { }
