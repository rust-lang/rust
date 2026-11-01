//! Regression test for issue #374, where previously rustc performed conditional jumps or moves that
//! incorrectly depended on uninitialized values.
//!
//! Issue: <https://github.com/rust-lang/rust/issues/374>.

//@ run-pass

#![allow(dead_code)]

enum TyS {
    Nil,
}

struct RawT {
    struct_: TyS,
    cname: Option<String>,
    hash: usize,
}

fn mk_raw_ty(st: TyS, cname: Option<String>) -> RawT {
    return RawT { struct_: st, cname: cname, hash: 0 };
}

pub fn main() {
    mk_raw_ty(TyS::Nil, None::<String>);
}
