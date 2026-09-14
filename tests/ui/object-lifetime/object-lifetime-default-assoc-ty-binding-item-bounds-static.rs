// Check that assoc item bindings correctly induce trait object lifetime defaults `'static` if the
// the trait & assoc ty doesn't have any lifetime params & the assoc ty isn't bounded by a lifetime.
//
//@ check-pass

trait Foo {
    type Item: ?Sized;

    fn item(&self) -> Box<Self::Item> { loop {} }
}

trait Bar {}

impl<T> Foo for T {
    type Item = dyn Bar;
}

fn is_static<T>(_: T) where T: 'static {}

// We elaborate `dyn Bar` to `dyn Bar + 'static` since the assoc ty isn't bounded by any lifetime.
// Notably, we don't elaborate it to `dyn Bar + 'r` since the trait object lifetime default induced
// by `Foo` (i.e., `'static`) shadows the one induced by `&` (`'r`).
fn bar<'r>(x: &'r str) -> &'r dyn Foo<Item = dyn Bar> { &() }

fn main() {
    let s = format!("foo");
    let r = bar(&s);

    is_static(r.item());
}
