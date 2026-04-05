// regression test for <https://github.com/rust-lang/rust/issues/153362>.
//@ check-pass

#![feature(unsafe_binders)]
#![allow(incomplete_features)]

struct ThinDst {
    b: unsafe<> (),
}

const fn t<const N: usize>(x: &[u8; N]) -> &ThinDst {
    unsafe { std::mem::transmute(x.as_ptr()) }
}

const C1: &ThinDst = t(b"d");

fn main() {}
