//@ run-pass
// Test that we can quantify lifetimes outside a constraint (i.e., including
// the self type) in a where clause.

// FIXME(static_mut_refs): this could use an atomic
#![allow(static_mut_refs)]

static mut COUNT: u32 = 1;

trait Bar<'a> {
    fn bar(&self);
}

trait Baz<'a>
{
    fn baz(&self);
}

impl<'a, 'b> Bar<'b> for &'a u32 {
    fn bar(&self) {
        unsafe { COUNT *= 2; }
    }
}

impl<'a, 'b> Baz<'b> for &'a u32 {
    fn baz(&self) {
        unsafe { COUNT *= 3; }
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
    unsafe {
        assert_eq!(COUNT, 12);
    }
}
