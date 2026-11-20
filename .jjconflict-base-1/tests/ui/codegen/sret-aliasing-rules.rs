//! Check that functions with sret results don't violate aliasing rules.
//!
//! When `foo = func(&mut foo)` is called, the compiler must avoid creating
//! two mutable references to the same variable simultaneously (one for the
//! parameter and one for the hidden sret out-pointer).
//!
//! Regression test for <https://github.com/rust-lang/rust/pull/18250>.

//@ run-pass

#[derive(Copy, Clone)]
pub struct Foo {
    f1: isize,
    _f2: isize,
}

#[inline(never)]
pub fn foo(f: &mut Foo) -> Foo {
    let ret = *f;
    f.f1 = 0;
    ret
}

pub fn main() {
    let mut f = Foo { f1: 8, _f2: 9 };
    f = foo(&mut f);
    assert_eq!(f.f1, 8);
}
