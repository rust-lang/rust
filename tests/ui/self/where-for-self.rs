//@ run-pass
// Test that we can quantify lifetimes outside a constraint (i.e., including
// the self type) in a where clause.

use std::sync::atomic::{AtomicUsize, Ordering};

static COUNT: AtomicUsize = AtomicUsize::new(1);

trait Bar<'a> {
    fn bar(&self);
}

trait Baz<'a>
{
    fn baz(&self);
}

impl<'a, 'b> Bar<'b> for &'a u32 {
    fn bar(&self) {
        COUNT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |mut c| {
            c *= 2;
            Some(c)
        }).unwrap();
    }
}

impl<'a, 'b> Baz<'b> for &'a u32 {
    fn baz(&self) {
        COUNT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |mut c| {
            c *= 3;
            Some(c)
        }).unwrap();
    }
}

// Test we can use the syntax for HRL including the self type.
fn foo1<T>(x: &T)
    where for<'a, 'b> &'a T: Bar<'b>
{
    x.bar()
}

// Test we can quantify multiple bounds (i.e., the precedence is sensible).
fn foo2<T>(x: &T)
    where for<'a, 'b> &'a T: Bar<'b> + Baz<'b>
{
    x.baz();
    x.bar()
}

fn main() {
    let x = 42;
    foo1(&x);
    foo2(&x);
    assert_eq!(COUNT.load(Ordering::Relaxed), 12);
}
