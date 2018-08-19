// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z parse-only

// Issue #50636

struct S {
    foo: u32 //~ expected `,`, or `}`, found `bar`
    //     ~^ HELP try adding a comma: ','
    bar: u32
}

fn main() {
    let s = S { foo: 5, bar: 6 };
}
