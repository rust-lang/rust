mod foo {
    pub struct Foo;
    impl Foo {
        fn bar(&self) {}
    }

    pub trait Baz {
        fn bar(&self) -> bool { true }
    }
    impl Baz for Foo {}
}

fn main() {
    use foo::Baz;

    // Check that `bar` resolves to the trait method, not the inherent impl method.
    let _: () = foo::Foo.bar(); //~ ERROR mismatched types
}
