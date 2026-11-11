// Regression test for #70813 (this used to trigger a debug assertion)

trait Trait {}

struct S;

impl<'a> Trait for &'a mut S {}

fn foo<X>(_: X)
where
    for<'b> &'b X: Trait,
{
}

fn main() {
    let s = S;
    foo::<S>(s); //~ ERROR the trait bound `for<'b> &'b S: Trait` is not satisfied
}
