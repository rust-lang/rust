// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo {
    x: int,
    y: int,
}

impl Index<int,int> for Foo {
    fn index(&self, z: &int) -> &int {
        if *z == 0 {
            &self.x
        } else {
            &self.y
        }
    }
}

impl IndexMut<int,int> for Foo {
    fn index_mut(&mut self, z: &int) -> &mut int {
        if *z == 0 {
            &mut self.x
        } else {
            &mut self.y
        }
    }
}

fn main() {
    let mut f = Foo {
        x: 1,
        y: 2,
    };
    assert_eq!(f[1], 2);
    f[0] = 3;
    assert_eq!(f[0], 3);
    {
        let p = &mut f[1];
        *p = 4;
    }
    {
        let p = &f[1];
        assert_eq!(*p, 4);
    }
}

