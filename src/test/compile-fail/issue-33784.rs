// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ops::Deref;

struct Obj<F> where F: FnMut() -> u32 {
    fn_ptr: fn() -> (),
    closure: F,
}

struct C {
    c_fn_ptr: fn() -> (),
}

struct D(C);

impl Deref for D {
    type Target = C;
    fn deref(&self) -> &C {
        &self.0
    }
}


fn empty() {}

fn main() {
    let o = Obj { fn_ptr: empty, closure: || 42 };
    let p = &o;
    p.closure(); //~ ERROR no method named `closure` found
    //~^ NOTE use `(p.closure)(...)` if you meant to call the function stored in the `closure` field
    let q = &p;
    q.fn_ptr(); //~ ERROR no method named `fn_ptr` found
    //~^ NOTE use `(q.fn_ptr)(...)` if you meant to call the function stored in the `fn_ptr` field
    let r = D(C { c_fn_ptr: empty });
    let s = &r;
    s.c_fn_ptr(); //~ ERROR no method named `c_fn_ptr` found
    //~^ NOTE use `(s.c_fn_ptr)(...)` if you meant to call the function stored in the `c_fn_ptr`
}
