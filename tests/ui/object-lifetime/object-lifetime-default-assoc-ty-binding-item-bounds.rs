// Ideally, given an assoc type binding `dyn Trait<AssocTy = Ty>`, we'd factor in the item bounds of
// assoc type `AssocTy` when computing the ambient object lifetime default for type `Ty`.
//
// However, since the current implementation can't handle this we instead conservatively and hackily
// treat the ambient object lifetime default of the RHS as indeterminate if any lifetime arguments
// are passed to the trait ref (or the GAT) thus rejecting any hidden object lifetime bounds.
// This way, we can still implement the desired behavior in the future.

trait Foo<'a> {
    type Item: ?Sized;

    fn item(&self) -> Box<Self::Item> { panic!() }
}

trait Bar {}

impl<T> Foo<'_> for T {
    type Item = dyn Bar;
}

fn is_static<T>(_: T) where T: 'static {}

// FIXME: Ideally, we'd elaborate `dyn Bar` to `dyn Bar + 'static` instead of rejecting it.
fn bar<'a>(x: &'a str) -> &'a dyn Foo<'a, Item = dyn Bar> { &() }
//~^ ERROR please supply an explicit bound

// FIXME: Ideally, we'd elaborate `dyn Bar` to `dyn Bar + 'static` instead of rejecting it.
fn baz(x: &str) -> &dyn Foo<Item = dyn Bar> { &() }
//~^ ERROR please supply an explicit bound

fn main() {
    let s = format!("foo");
    let r = bar(&s);
    is_static(r.item());
    let r = baz(&s);
    is_static(r.item());
}
