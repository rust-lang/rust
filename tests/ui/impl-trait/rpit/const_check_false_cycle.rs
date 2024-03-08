//! This test causes a cycle error when checking whether the
//! return type is `Freeze` during const checking, even though
//! the information is readily available.

const fn f() -> impl Eq {
    //~^ ERROR cycle detected
    g()
}
const fn g() {}

fn main() {}
