// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct A;

impl A {
    fn take_mutably(&mut self) {}
}

fn identity<T>(t: T) -> T {
    t
}

// Issue 46095
// Built-in indexing should be used even when the index is not
// trivially an integer
// Overloaded indexing would cause wrapped to be borrowed mutably

fn main() {
    let mut a1 = A;
    let mut a2 = A;

    let wrapped = [&mut a1, &mut a2];

    {
        wrapped[0 + 1 - 1].take_mutably();
    }

    {
        wrapped[identity(0)].take_mutably();
    }
}
