//@ run-pass
// Test calling methods on an impl for a bare trait.

//@ aux-build:traitimpl.rs

use std::sync::atomic::{AtomicUsize, Ordering};

extern crate traitimpl;
use traitimpl::Bar;

static COUNT: AtomicUsize = AtomicUsize::new(1);

trait T {
    fn t(&self) {} //~ WARN method `t` is never used
}

impl<'a> dyn T+'a {
    fn foo(&self) {
        COUNT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |mut c| {
            c *= 2;
            Some(c)
        }).unwrap();
    }
    fn bar() {
        COUNT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |mut c| {
            c *= 3;
            Some(c)
        }).unwrap();
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
