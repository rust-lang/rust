// run-rustfix
// edition:2018

#![warn(clippy::use_self)]
#![allow(dead_code)]
#![allow(clippy::should_implement_trait)]

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

mod lifetimes {
    struct Foo<'a> {
        foo_str: &'a str,
    }

    impl<'a> Foo<'a> {
        // Cannot use `Self` as return type, because the function is actually `fn foo<'b>(s: &'b str) ->
        // Foo<'b>`
        fn foo(s: &str) -> Foo {
            Foo { foo_str: s }
        }
        // cannot replace with `Self`, because that's `Foo<'a>`
        fn bar() -> Foo<'static> {
            Foo { foo_str: "foo" }
        }

        // FIXME: the lint does not handle lifetimed struct
        // `Self` should be applicable here
        fn clone(&self) -> Foo<'a> {
            Foo { foo_str: self.foo_str }
        }
    }
}

mod issue2894 {
    trait IntoBytes {
        #[allow(clippy::wrong_self_convention)]
        fn into_bytes(&self) -> Vec<u8>;
    }

    // This should not be linted
    impl IntoBytes for u8 {
        fn into_bytes(&self) -> Vec<u8> {
            vec![*self]
        }
    }
}

mod existential {
    struct Foo;

    impl Foo {
        fn bad(foos: &[Self]) -> impl Iterator<Item = &Foo> {
            foos.iter()
        }

        fn good(foos: &[Self]) -> impl Iterator<Item = &Self> {
            foos.iter()
        }
    }
}

mod tuple_structs {
    pub struct TS(i32);

    impl TS {
        pub fn ts() -> Self {
            TS(0)
        }
    }
}

mod macros {
    macro_rules! use_self_expand {
        () => {
            fn new() -> Foo {
                Foo {}
            }
        };
    }

    struct Foo {}

    impl Foo {
        use_self_expand!(); // Should lint in local macros
    }
}

mod nesting {
    struct Foo {}
    impl Foo {
        fn foo() {
            #[allow(unused_imports)]
            use self::Foo; // Can't use Self here
            struct Bar {
                foo: Foo, // Foo != Self
            }

            impl Bar {
                fn bar() -> Bar {
                    Bar { foo: Foo {} }
                }
            }

            // Can't use Self here
            fn baz() -> Foo {
                Foo {}
            }
        }

        // Should lint here
        fn baz() -> Foo {
            Foo {}
        }
    }

    enum Enum {
        A,
        B(u64),
        C { field: bool },
    }
    impl Enum {
        fn method() {
            #[allow(unused_imports)]
            use self::Enum::*; // Issue 3425
            static STATIC: Enum = Enum::A; // Can't use Self as type
        }

        fn method2() {
            let _ = Enum::B(42);
            let _ = Enum::C { field: true };
            let _ = Enum::A;
        }
    }
}

mod issue3410 {

    struct A;
    struct B;

    trait Trait<T> {
        fn a(v: T);
    }

    impl Trait<Vec<A>> for Vec<B> {
        fn a(_: Vec<A>) {}
    }
}

#[allow(clippy::no_effect, path_statements)]
mod rustfix {
    mod nested {
        pub struct A {}
    }

    impl nested::A {
        const A: bool = true;

        fn fun_1() {}

        fn fun_2() {
            nested::A::fun_1();
            nested::A::A;

            nested::A {};
        }
    }
}

mod issue3567 {
    struct TestStruct {}
    impl TestStruct {
        fn from_something() -> Self {
            Self {}
        }
    }

    trait Test {
        fn test() -> TestStruct;
    }

    impl Test for TestStruct {
        fn test() -> TestStruct {
            TestStruct::from_something()
        }
    }
}

mod paths_created_by_lowering {
    use std::ops::Range;

    struct S {}

    impl S {
        const A: usize = 0;
        const B: usize = 1;

        async fn g() -> S {
            S {}
        }

        fn f<'a>(&self, p: &'a [u8]) -> &'a [u8] {
            &p[S::A..S::B]
        }
    }

    trait T {
        fn f<'a>(&self, p: &'a [u8]) -> &'a [u8];
    }

    impl T for Range<u8> {
        fn f<'a>(&self, p: &'a [u8]) -> &'a [u8] {
            &p[0..1]
        }
    }
}
