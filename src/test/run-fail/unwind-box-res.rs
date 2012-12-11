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

fn failfn() {
    fail;
}

struct r {
  v: *int,
}

impl r : Drop {
    fn finalize(&self) {
        unsafe {
            let _v2: ~int = cast::reinterpret_cast(&self.v);
        }
    }
}

fn r(v: *int) -> r {
    r {
        v: v
    }
}

fn main() unsafe {
    let i1 = ~0;
    let i1p = cast::reinterpret_cast(&i1);
    cast::forget(move i1);
    let x = @r(i1p);
    failfn();
    log(error, x);
}
