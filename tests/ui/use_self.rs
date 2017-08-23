#![feature(plugin)]
#![plugin(clippy)]
#![warn(use_self)]
#![allow(dead_code)]
#![allow(should_implement_trait)]


fn main() {}

mod use_self {
    struct Foo {}

    impl Foo {
        fn new() -> Foo {
            Foo {}
        }
        fn test() -> Foo {
            Foo::new()
        }
    }

    impl Default for Foo {
        fn default() -> Foo {
            Foo::new()
        }
    }
}

mod better {
    struct Foo {}

    impl Foo {
        fn new() -> Self {
            Self {}
        }
        fn test() -> Self {
            Self::new()
        }
    }

    impl Default for Foo {
        fn default() -> Self {
            Self::new()
        }
    }
}

//todo the lint does not handle lifetimed struct
//the following module should trigger the lint on the third method only
mod lifetimes {
    struct Foo<'a>{foo_str: &'a str}

    impl<'a> Foo<'a> {
        // Cannot use `Self` as return type, because the function is actually `fn foo<'b>(s: &'b str) -> Foo<'b>`
        fn foo(s: &str) -> Foo {
            Foo { foo_str: s }
        }
        // cannot replace with `Self`, because that's `Foo<'a>`
        fn bar() -> Foo<'static> {
            Foo { foo_str: "foo"}
        }

        // `Self` is applicable here
        fn clone(&self) -> Foo<'a> {
            Foo {foo_str: self.foo_str}
        }
    }
}
