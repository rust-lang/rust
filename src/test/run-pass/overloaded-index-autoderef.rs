// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test overloaded indexing combined with autoderef.

#![allow(unknown_features)]
#![feature(box_syntax)]

use std::ops::{Index, IndexMut};

struct Foo {
    x: int,
    y: int,
}

impl Index<int> for Foo {
    type Output = int;

    fn index(&self, z: &int) -> &int {
        if *z == 0 {
            &self.x
        } else {
            &self.y
        }
    }
}

impl IndexMut<int> for Foo {
    type Output = int;

    fn index_mut(&mut self, z: &int) -> &mut int {
        if *z == 0 {
            &mut self.x
        } else {
            &mut self.y
        }
    }
}

trait Int {
    fn get(self) -> int;
    fn get_from_ref(&self) -> int;
    fn inc(&mut self);
}

impl Int for int {
    fn get(self) -> int { self }
    fn get_from_ref(&self) -> int { *self }
    fn inc(&mut self) { *self += 1; }
}

fn main() {
    let mut f = box Foo {
        x: 1,
        y: 2,
    };

    assert_eq!(f[1], 2);

    f[0] = 3;

    assert_eq!(f[0], 3);

    // Test explicit IndexMut where `f` must be autoderef:
    {
        let p = &mut f[1];
        *p = 4;
    }

    // Test explicit Index where `f` must be autoderef:
    {
        let p = &f[1];
        assert_eq!(*p, 4);
    }

    // Test calling methods with `&mut self`, `self, and `&self` receivers:
    f[1].inc();
    assert_eq!(f[1].get(), 5);
    assert_eq!(f[1].get_from_ref(), 5);
}
