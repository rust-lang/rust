pub struct Foo;

mod bar {
    struct Foo;

    mod baz {
        use *;
        use bar::*;
        fn f(_: Foo) {} //~ ERROR `Foo` is ambiguous
    }
}

fn main() {}
