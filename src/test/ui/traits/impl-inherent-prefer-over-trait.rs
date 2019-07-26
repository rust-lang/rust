// run-pass

struct Foo;

trait Trait {
    fn bar(&self);
}

// Inherent impls should be preferred over trait ones.
impl Foo {
    fn bar(&self) {}
}

impl dyn Trait {
    fn baz(_: &Foo) {}
}

impl Trait for Foo {
    fn bar(&self) { panic!("wrong method called!") }
}

fn main() {
    Foo.bar();
    Foo::bar(&Foo);
    <Foo>::bar(&Foo);

    // Should work even if Trait::baz doesn't exist.
    // N.B: `<Trait>::bar` would be ambiguous.
    <dyn Trait>::baz(&Foo);
}
