// compile-flags: -Ztrait-solver=next
// check-pass

trait Trait<'a> {
    type Item: for<'b> Trait2<'b>;
}

trait Trait2<'a> {}
impl Trait2<'_> for () {}

fn needs_trait(_: Box<impl for<'a> Trait<'a> + ?Sized>) {}

fn foo(x: Box<dyn for<'a> Trait<'a, Item = ()>>) {
    needs_trait(x);
}

fn main() {}
