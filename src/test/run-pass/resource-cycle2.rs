// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Don't leak the unique pointers

use std::cast;

struct U {
    a: int,
    b: int,
    c: *int
}

struct r {
  v: U,
}

impl Drop for r {
    fn drop(&self) {
        unsafe {
            let _v2: ~int = cast::transmute(self.v.c);
        }
    }
}

fn r(v: U) -> r {
    r {
        v: v
    }
}

struct t(Node);

struct Node {
    next: Option<@mut t>,
    r: r
}

pub fn main() {
    unsafe {
        let i1 = ~0xA;
        let i1p = cast::transmute_copy(&i1);
        cast::forget(i1);
        let i2 = ~0xA;
        let i2p = cast::transmute_copy(&i2);
        cast::forget(i2);

        let u1 = U {a: 0xB, b: 0xC, c: i1p};
        let u2 = U {a: 0xB, b: 0xC, c: i2p};

        let x1 = @mut t(Node {
            next: None,
            r: r(u1)
        });
        let x2 = @mut t(Node {
            next: None,
            r: r(u2)
        });
        x1.next = Some(x2);
        x2.next = Some(x1);
    }
}
