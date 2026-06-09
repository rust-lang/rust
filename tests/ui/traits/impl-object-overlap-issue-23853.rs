//@ run-pass
// Test that we are able to compile the case where both a blanket impl
// and the object type itself supply the required trait obligation.
// In this case, the blanket impl for `Foo` applies to any type,
// including `Bar`, but the object type `Bar` also implicitly supplies
// this context.

trait Foo { fn dummy(&self) { } } //~ WARN method `dummy` is never used

trait Bar: Foo { }

impl<T:?Sized> Foo for T { }

fn want_foo<B:?Sized+Foo>() { }

fn main() {
    want_foo::<dyn Bar>();
}
