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

struct r {
  v: *int,
}

impl Drop for r {
    fn drop(&self) {
        unsafe {
            info!("r's dtor: self = %x, self.v = %x, self.v's value = %x",
              cast::transmute::<*r, uint>(self),
              cast::transmute::<**int, uint>(&(self.v)),
              cast::transmute::<*int, uint>(self.v));
            let v2: ~int = cast::transmute(self.v);
        }
    }
}

fn r(v: *int) -> r {
    unsafe {
        r {
            v: v
        }
    }
}

struct t(Node);

struct Node {
    next: Option<@mut t>,
    r: r
}

pub fn main() {
    unsafe {
        let i1 = ~0;
        let i1p = cast::transmute_copy(&i1);
        cast::forget(i1);
        let i2 = ~0;
        let i2p = cast::transmute_copy(&i2);
        cast::forget(i2);

        let mut x1 = @mut t(Node{
            next: None,
              r: {
              let rs = r(i1p);
              info!("r = %x", cast::transmute::<*r, uint>(&rs));
              rs }
        });

        info!("x1 = %x, x1.r = %x",
               cast::transmute::<@mut t, uint>(x1),
               cast::transmute::<*r, uint>(&x1.r));

        let mut x2 = @mut t(Node{
            next: None,
              r: {
              let rs = r(i2p);
              info!("r2 = %x", cast::transmute::<*r, uint>(&rs));
              rs
                }
        });

        info!("x2 = %x, x2.r = %x",
               cast::transmute::<@mut t, uint>(x2),
               cast::transmute::<*r, uint>(&(x2.r)));

        x1.next = Some(x2);
        x2.next = Some(x1);
    }
}
