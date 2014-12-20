// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn require_copy<T: Copy>(x: T) {}

struct Bar<T> { x: T }

trait Foo<T> {
    fn needs_copy(self) where T: Copy;
    fn fails_copy(self);
}

// Ensure constraints are only attached to methods locally
impl<T> Foo<T> for Bar<T> {
    fn needs_copy(self) where T: Copy {
        require_copy(self.x);

    }

    fn fails_copy(self) {
        require_copy(self.x);
        //~^ ERROR the trait `core::kinds::Copy` is not implemented for the type `T`
    }
}

fn main() {}
