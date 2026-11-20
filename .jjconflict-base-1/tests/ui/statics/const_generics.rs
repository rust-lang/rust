//! Check that we lose the information that `BAR` points to `FOO`
//! when going through a const generic.
//! This is not an intentional guarantee, it just describes the status quo.

//@ run-pass
//@ ignore-backends: gcc
// With optimizations, LLVM will deduplicate the constant `X` whose
// value is `&42` to just be a reference to the static. This is correct,
// but obscures the issue we're trying to show.
//@ revisions: opt noopt
//@[noopt] compile-flags: -Copt-level=0
//@[opt] compile-flags: -O

#![feature(adt_const_params, unsized_const_params)]
#![allow(incomplete_features)]

static FOO: usize = 42;
const BAR: &usize = &FOO;
fn foo<const X: &'static usize>() {
    // Without optimizations, `X` ends up pointing to a copy of `FOO` instead of `FOO` itself.
    assert_eq!(cfg!(opt), std::ptr::eq(X, &FOO));
}

fn main() {
    foo::<BAR>();
}
