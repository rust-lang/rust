// Test that `dyn Bar<Item = XX>` uses `'static` as the default object
// lifetime bound for the type `XX`.

trait Foo<'a> {
    type Item: ?Sized;

    fn item(&self) -> Box<Self::Item> { panic!() }
}

trait Bar { }

impl<T> Foo<'_> for T {
    type Item = dyn Bar;
}

fn is_static<T>(_: T) where T: 'static { }

// Here, we should default to `dyn Bar + 'static`, but the current
// code forces us into a conservative, hacky path.
fn bar<'a>(x: &'a str) -> &'a dyn Foo<'a, Item = dyn Bar> { &() }
//~^ ERROR please supply an explicit bound

fn main() {
    let s = format!("foo");
    let r = bar(&s);
    is_static(r.item());
}
