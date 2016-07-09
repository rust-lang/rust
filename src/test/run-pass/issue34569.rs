// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-g

// In this test we just want to make sure that the code below does not lead to
// a debuginfo verification assertion during compilation. This was caused by the
// closure in the guard being translated twice due to how match expressions are
// handled.
//
// See https://github.com/rust-lang/rust/issues/34569 for details.

fn main() {
    match 0 {
        e if (|| { e == 0 })() => {},
        1 => {},
        _ => {}
    }
}
