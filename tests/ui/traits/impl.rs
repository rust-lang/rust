//@ run-pass
// Test calling methods on an impl for a bare trait.

//@ aux-build:traitimpl.rs

extern crate traitimpl;
use std::sync::atomic::{AtomicUsize, Ordering};
use traitimpl::Bar;

static COUNT: AtomicUsize = AtomicUsize::new(1);

fn multiply_count(factor: usize) {
    COUNT.try_update(Ordering::Relaxed, Ordering::Relaxed, |count| Some(count * factor)).unwrap();
}

trait T {
    fn t(&self) {} //~ WARN method `t` is never used
}

impl<'a> dyn T+'a {
    fn foo(&self) {
        multiply_count(2);
    }
    fn bar() {
        multiply_count(3);
    }
}

impl T for isize {}

struct Foo;
impl<'a> Bar<'a> for Foo {}

fn main() {
    let x: &dyn T = &42;

    x.foo();
    <dyn T>::foo(x);
    <dyn T>::bar();

    assert_eq!(COUNT.load(Ordering::Relaxed), 12);

    // Cross-crait case
    let x: &dyn Bar = &Foo;
    x.bar();
}
