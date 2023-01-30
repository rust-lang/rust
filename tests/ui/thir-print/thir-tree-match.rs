// check-pass
// compile-flags: -Zunpretty=thir-tree

enum Bar {
    First,
    Second,
    Third,
}

enum Foo {
    FooOne(Bar),
    FooTwo,
}

fn has_match(foo: Foo) -> bool {
    match foo {
        Foo::FooOne(Bar::First) => true,
        Foo::FooOne(_) => false,
        Foo::FooTwo => true,
    }
}

fn main() {}
