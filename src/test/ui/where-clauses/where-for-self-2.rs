// Test that we can quantify lifetimes outside a constraint (i.e., including
// the self type) in a where clause. Specifically, test that implementing for a
// specific lifetime is not enough to satisfy the `for<'a> ...` constraint, which
// should require *all* lifetimes.

static X: &'static u32 = &42;

trait Bar {
    fn bar(&self);
}

impl Bar for &'static u32 {
    fn bar(&self) {}
}

fn foo<T>(x: &T)
    where for<'a> &'a T: Bar
{}

fn main() {
    foo(&X); //~ ERROR trait bound
}
