//! check that we lose the information that `BAR` points to `FOO`
//! when going through a const generic.

// With optimizations, LLVM will deduplicate the constant `X` whose
// value is `&42` to just be a reference to the static. This is correct,
// but obscures the issue we're trying to show.
//@ compile-flags: -Copt-level=0

#![feature(const_refs_to_static)]
#![feature(adt_const_params)]
#![allow(incomplete_features)]

static FOO: usize = 42;
const BAR: &usize = &FOO;
fn foo<const X: &'static usize>() {
    assert!(!std::ptr::eq(X, &FOO));
}

fn main() {
    foo::<BAR>();
    //~^ ERROR: encountered a reference pointing to a static variable in a constant
}
