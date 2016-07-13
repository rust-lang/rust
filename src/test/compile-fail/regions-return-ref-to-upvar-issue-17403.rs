// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that closures cannot subvert aliasing restrictions

fn main() {
    // Unboxed closure case
    {
        let mut x = 0;
        let mut f = || &mut x; //~ ERROR cannot infer
        let x = f();
        let y = f();
    }
}
