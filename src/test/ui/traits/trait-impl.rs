// run-pass
// Test calling methods on an impl for a bare trait.

// aux-build:traitimpl.rs

extern crate traitimpl;
use traitimpl::Bar;

static mut COUNT: usize = 1;

trait T {
    fn t(&self) {}
}

impl<'a> dyn T+'a {
    fn foo(&self) {
        unsafe { COUNT *= 2; }
    }
    fn bar() {
        unsafe { COUNT *= 3; }
    }
}

impl T for isize {}

struct Foo;
impl<'a> Bar<'a> for Foo {}

fn main() {
    let x: &dyn T = &42;

    x.foo();
    T::foo(x);
    T::bar();

    unsafe { assert_eq!(COUNT, 12); }

    // Cross-crait case
    let x: &dyn Bar = &Foo;
    x.bar();
}
