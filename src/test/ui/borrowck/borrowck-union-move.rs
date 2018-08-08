// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(untagged_unions)]

#[derive(Clone, Copy)]
struct Copy;
struct NonCopy;

union Unn {
    n1: NonCopy,
    n2: NonCopy,
}
union Ucc {
    c1: Copy,
    c2: Copy,
}
union Ucn {
    c: Copy,
    n: NonCopy,
}

fn main() {
    unsafe {
        // 2 NonCopy
        {
            let mut u = Unn { n1: NonCopy };
            let a = u.n1;
            let a = u.n1; //~ ERROR use of moved value: `u.n1`
        }
        {
            let mut u = Unn { n1: NonCopy };
            let a = u.n1;
            let a = u; //~ ERROR use of partially moved value: `u`
        }
        {
            let mut u = Unn { n1: NonCopy };
            let a = u.n1;
            let a = u.n2; //~ ERROR use of moved value: `u.n2`
        }
        // 2 Copy
        {
            let mut u = Ucc { c1: Copy };
            let a = u.c1;
            let a = u.c1; // OK
        }
        {
            let mut u = Ucc { c1: Copy };
            let a = u.c1;
            let a = u; // OK
        }
        {
            let mut u = Ucc { c1: Copy };
            let a = u.c1;
            let a = u.c2; // OK
        }
        // 1 Copy, 1 NonCopy
        {
            let mut u = Ucn { c: Copy };
            let a = u.c;
            let a = u.c; // OK
        }
        {
            let mut u = Ucn { c: Copy };
            let a = u.n;
            let a = u.n; //~ ERROR use of moved value: `u.n`
        }
        {
            let mut u = Ucn { c: Copy };
            let a = u.n;
            let a = u.c; //~ ERROR use of moved value: `u.c`
        }
        {
            let mut u = Ucn { c: Copy };
            let a = u.c;
            let a = u.n; // OK
        }
        {
            let mut u = Ucn { c: Copy };
            let a = u.c;
            let a = u; // OK
        }
        {
            let mut u = Ucn { c: Copy };
            let a = u.n;
            let a = u; //~ ERROR use of partially moved value: `u`
        }
    }
}
