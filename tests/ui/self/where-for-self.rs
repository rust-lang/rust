//@ run-pass
// Test that we can quantify lifetimes outside a constraint (i.e., including
// the self type) in a where clause.

use std::sync::atomic::{AtomicU32, Ordering};

static COUNT: AtomicU32 = AtomicU32::new(1);

fn multiply_count(factor: u32) {
    COUNT.try_update(Ordering::Relaxed, Ordering::Relaxed, |count| Some(count * factor)).unwrap();
}

trait Bar<'a> {
    fn bar(&self);
}

trait Baz<'a>
{
    fn baz(&self);
}

impl<'a, 'b> Bar<'b> for &'a u32 {
    fn bar(&self) {
        multiply_count(2);
    }
}

impl<'a, 'b> Baz<'b> for &'a u32 {
    fn baz(&self) {
        multiply_count(3);
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
