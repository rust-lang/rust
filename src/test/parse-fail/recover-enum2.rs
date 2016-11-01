// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z parse-only -Z continue-parse-after-error

fn main() {
    enum Test {
        Var1,
        Var2(String),
        Var3 {
            abc: {}, //~ ERROR: expected type, found `{`
        },
    }

    // recover...
    let a = 1;
    enum Test2 {
        Fine,
    }

    enum Test3 {
        StillFine {
            def: i32,
        },
    }

    {
        // fail again
        enum Test4 {
            Nope(i32 {}) //~ ERROR: found `{`
                         //~^ ERROR: found `{`
        }
    }
    // still recover later
    let bad_syntax = _; //~ ERROR: found `_`
}
