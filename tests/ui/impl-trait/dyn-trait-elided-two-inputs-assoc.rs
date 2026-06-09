// Test that we don't get an error with `dyn Bar` in an impl Trait
// when there are multiple inputs.  The `dyn Bar` should default to `+
// 'static`. This used to erroneously generate an error (cc #62517).
//
//@ check-pass

trait Foo { type Item: ?Sized; }
trait Bar { }

impl<T> Foo for T {
    type Item = dyn Bar;
}

fn foo(x: &str, y: &str) -> impl Foo<Item = dyn Bar> { () }

fn main() { }
