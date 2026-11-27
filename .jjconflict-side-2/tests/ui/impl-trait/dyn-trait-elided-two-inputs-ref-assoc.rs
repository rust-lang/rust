// Test that we don't get an error with `dyn Bar` in an impl Trait
// when there are multiple inputs.  The `dyn Bar` should default to `+
// 'static`. This used to erroneously generate an error (cc #62517).
//
//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

trait Foo {
    type Item: ?Sized;

    fn item(&self) -> Box<Self::Item> { panic!() }
}

trait Bar { }

impl<T> Foo for T {
    type Item = dyn Bar;
}

fn is_static<T>(_: T) where T: 'static { }

fn bar(x: &str) -> &impl Foo<Item = dyn Bar> { &() }

fn main() {
    let s = format!("foo");
    let r = bar(&s);
    is_static(r.item());
}
