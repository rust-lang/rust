//! This test causes a cycle error when checking whether the
//! return type is `Freeze` during const checking, even though
//! the information is readily available.

//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@[next] check-pass

const fn f() -> impl Eq {
    //[current]~^ ERROR cycle detected
    g()
}
const fn g() {}

fn main() {}
