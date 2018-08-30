// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// "guessing" in trait selection can affect `copy_or_move`. Check that this
// is correctly handled. I am not sure what is the "correct" behaviour,
// but we should at least not ICE.

use std::mem;

struct U([u8; 1337]);

struct S<'a,T:'a>(&'a T);
impl<'a, T> Clone for S<'a, T> { fn clone(&self) -> Self { S(self.0) } }
/// This impl triggers inference "guessing" - S<_>: Copy => _ = U
impl<'a> Copy for S<'a, Option<U>> {}

fn assert_impls_fn<R,T: Fn()->R>(_: &T){}

fn main() {
    let n = None;
    let e = S(&n);
    let f = || {
        // S being copy is critical for this to work
        drop(e);
        mem::size_of_val(e.0)
    };
    assert_impls_fn(&f);
    assert_eq!(f(), 1337+1);

    assert_eq!((|| {
        // S being Copy is not critical here, but
        // we check it anyway.
        let n = None;
        let e = S(&n);
        let ret = mem::size_of_val(e.0);
        drop(e);
        ret
    })(), 1337+1);
}
