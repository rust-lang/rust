//! This test caused a cycle error when checking whether the
//! return type is `Freeze` during const checking, even though
//! the information is readily available.

//@ check-pass

const fn f() -> impl Eq {
    g()
}
const fn g() {}

fn main() {}
