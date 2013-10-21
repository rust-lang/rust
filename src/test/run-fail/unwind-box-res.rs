// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:fail

use std::cast;

fn failfn() {
    fail!();
}

struct r {
  v: *int,
}

impl Drop for r {
    fn drop(&mut self) {
        unsafe {
            let _v2: ~int = cast::transmute(self.v);
        }
    }
}

fn r(v: *int) -> r {
    r {
        v: v
    }
}

fn main() {
    unsafe {
        let i1 = ~0;
        let i1p = cast::transmute_copy(&i1);
        cast::forget(i1);
        let x = @r(i1p);
        failfn();
        error!("{:?}", x);
    }
}
