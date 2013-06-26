// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test
// error-pattern: ran out of stack
struct R {
    b: int,
}

impl Drop for R {
    fn drop(&self) {
        let _y = R { b: self.b };
    }
}

fn main() {
    let _x = R { b: 0 };
}
