// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Obj<F> where F: FnMut() -> u32 {
    closure: F,
    nfn: usize,
}

struct S<F> where F: FnMut() -> u32 {
    v: Obj<F>,
}

fn func() -> u32 {
    0
}

fn main() {
    let o = Obj { closure: || 42, nfn: 42 };
    o.closure(); //~ ERROR no method named `closure` found
    //~^ NOTE use `(o.closure)(...)` if you meant to call the function stored in the `closure` field

    // TODO move these to a new test for #2392
    let x = o.nfn(); //~ ERROR no method named `nfn` found
    //~^ NOTE did you mean to write `o.nfn`?

    let b = Obj { closure: func, nfn: 5 };
    b.closure(); //~ ERROR no method named `closure` found
    //~^ NOTE use `(b.closure)(...)` if you meant to call the function stored in the `closure` field

    let s = S { v: b };
    s.v.closure();//~ ERROR no method named `closure` found
    //~^ NOTE use `(s.v.closure)(...)` if you meant to call the function stored in the `closure` field
    s.v.nfn();//~ ERROR no method named `nfn` found
    //~^ NOTE did you mean to write `s.v.nfn`?
}
