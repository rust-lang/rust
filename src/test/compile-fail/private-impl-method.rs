mod a {
    #[legacy_exports];
    struct Foo {
        x: int
    }

    impl Foo {
        priv fn foo() {}
    }
}

fn main() {
    let s = a::Foo { x: 1 };
    s.foo();    //~ ERROR method `foo` is private
}

