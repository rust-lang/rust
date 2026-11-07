#![allow(dead_code)]
#![allow(dropping_copy_types)]

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
    //~^ ERROR type annotations needed for `Option<_>`
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
