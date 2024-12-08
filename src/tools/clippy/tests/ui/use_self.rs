//@aux-build:proc_macro_derive.rs

#![warn(clippy::use_self)]
#![allow(dead_code, unreachable_code)]
#![allow(
    clippy::should_implement_trait,
    clippy::upper_case_acronyms,
    clippy::from_over_into,
    clippy::self_named_constructors,
    clippy::needless_lifetimes,
    clippy::missing_transmute_annotations
)]

#[macro_use]
extern crate proc_macro_derive;

fn main() {}

mod use_self {
    struct Foo;

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
    struct Foo;

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
    #[derive(Clone, Copy)]
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

        fn clone(&self) -> Foo<'a> {
            Foo { foo_str: self.foo_str }
        }

        // Cannot replace with `Self` because the lifetime is not `'a`.
        fn eq<'b>(&self, other: Foo<'b>) -> bool {
            let x: Foo<'_> = other;
            self.foo_str == other.foo_str
        }

        fn f(&self) -> Foo<'_> {
            *self
        }
    }
}

mod issue2894 {
    trait IntoBytes {
        fn to_bytes(self) -> Vec<u8>;
    }

    // This should not be linted
    impl IntoBytes for u8 {
        fn to_bytes(self) -> Vec<u8> {
            vec![self]
        }
    }
}

mod existential {
    struct Foo;

    impl Foo {
        fn bad(foos: &[Foo]) -> impl Iterator<Item = &Foo> {
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

    struct Foo;

    impl Foo {
        use_self_expand!(); // Should not lint in local macros
    }

    #[derive(StructAUseSelf)] // Should not lint in derives
    struct A;
}

mod nesting {
    struct Foo;
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
        fn a(v: T) -> Self;
    }

    impl Trait<Vec<A>> for Vec<B> {
        fn a(_: Vec<A>) -> Self {
            unimplemented!()
        }
    }

    impl<T> Trait<Vec<A>> for Vec<T>
    where
        T: Trait<B>,
    {
        fn a(v: Vec<A>) -> Self {
            <Vec<B>>::a(v).into_iter().map(Trait::a).collect()
        }
    }
}

#[allow(clippy::no_effect, path_statements)]
mod rustfix {
    mod nested {
        pub struct A;
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
    struct TestStruct;
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

    struct S;

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

// reused from #1997
mod generics {
    struct Foo<T> {
        value: T,
    }

    impl<T> Foo<T> {
        // `Self` is applicable here
        fn foo(value: T) -> Foo<T> {
            Foo::<T> { value }
        }

        // `Cannot` use `Self` as a return type as the generic types are different
        fn bar(value: i32) -> Foo<i32> {
            Foo { value }
        }
    }
}

mod issue4140 {
    pub struct Error<From, To> {
        _from: From,
        _too: To,
    }

    pub trait From<T> {
        type From;
        type To;

        fn from(value: T) -> Self;
    }

    pub trait TryFrom<T>
    where
        Self: Sized,
    {
        type From;
        type To;

        fn try_from(value: T) -> Result<Self, Error<Self::From, Self::To>>;
    }

    // FIXME: Suggested fix results in infinite recursion.
    // impl<F, T> TryFrom<F> for T
    // where
    //     T: From<F>,
    // {
    //     type From = Self::From;
    //     type To = Self::To;

    //     fn try_from(value: F) -> Result<Self, Error<Self::From, Self::To>> {
    //         Ok(From::from(value))
    //     }
    // }

    impl From<bool> for i64 {
        type From = bool;
        type To = Self;

        fn from(value: bool) -> Self {
            if value { 100 } else { 0 }
        }
    }
}

mod issue2843 {
    trait Foo {
        type Bar;
    }

    impl Foo for usize {
        type Bar = u8;
    }

    impl<T: Foo> Foo for Option<T> {
        type Bar = Option<T::Bar>;
    }
}

mod issue3859 {
    pub struct Foo;
    pub struct Bar([usize; 3]);

    impl Foo {
        pub const BAR: usize = 3;

        pub fn foo() {
            const _X: usize = Foo::BAR;
            // const _Y: usize = Self::BAR;
        }
    }
}

mod issue4305 {
    trait Foo: 'static {}

    struct Bar;

    impl Foo for Bar {}

    impl<T: Foo> From<T> for Box<dyn Foo> {
        fn from(t: T) -> Self {
            Box::new(t)
        }
    }
}

mod lint_at_item_level {
    struct Foo;

    #[allow(clippy::use_self)]
    impl Foo {
        fn new() -> Foo {
            Foo {}
        }
    }

    #[allow(clippy::use_self)]
    impl Default for Foo {
        fn default() -> Foo {
            Foo::new()
        }
    }
}

mod lint_at_impl_item_level {
    struct Foo;

    impl Foo {
        #[allow(clippy::use_self)]
        fn new() -> Foo {
            Foo {}
        }
    }

    impl Default for Foo {
        #[allow(clippy::use_self)]
        fn default() -> Foo {
            Foo::new()
        }
    }
}

mod issue4734 {
    #[repr(C, packed)]
    pub struct X {
        pub x: u32,
    }

    impl From<X> for u32 {
        fn from(c: X) -> Self {
            unsafe { core::mem::transmute(c) }
        }
    }
}

mod nested_paths {
    use std::convert::Into;
    mod submod {
        pub struct B;
        pub struct C;

        impl Into<C> for B {
            fn into(self) -> C {
                C {}
            }
        }
    }

    struct A<T> {
        t: T,
    }

    impl<T> A<T> {
        fn new<V: Into<T>>(v: V) -> Self {
            Self { t: Into::into(v) }
        }
    }

    impl A<submod::C> {
        fn test() -> Self {
            A::new::<submod::B>(submod::B {})
        }
    }
}

mod issue6818 {
    #[derive(serde::Deserialize)]
    struct A {
        a: i32,
    }
}

mod issue7206 {
    struct MyStruct<const C: char>;
    impl From<MyStruct<'a'>> for MyStruct<'b'> {
        fn from(_s: MyStruct<'a'>) -> Self {
            Self
        }
    }

    // keep linting non-`Const` generic args
    struct S<'a> {
        inner: &'a str,
    }

    struct S2<T> {
        inner: T,
    }

    impl<T> S2<T> {
        fn new() -> Self {
            unimplemented!();
        }
    }

    impl<'a> S2<S<'a>> {
        fn new_again() -> Self {
            S2::new()
        }
    }
}

mod self_is_ty_param {
    trait Trait {
        type Type;
        type Hi;

        fn test();
    }

    impl<I> Trait for I
    where
        I: Iterator,
        I::Item: Trait, // changing this to Self would require <Self as Iterator>
    {
        type Type = I;
        type Hi = I::Item;

        fn test() {
            let _: I::Item;
            let _: I; // this could lint, but is questionable
        }
    }
}

mod use_self_in_pat {
    enum Foo {
        Bar,
        Baz,
    }

    impl Foo {
        fn do_stuff(self) {
            match self {
                Foo::Bar => unimplemented!(),
                Foo::Baz => unimplemented!(),
            }
            match Some(1) {
                Some(_) => unimplemented!(),
                None => unimplemented!(),
            }
            if let Foo::Bar = self {
                unimplemented!()
            }
        }
    }
}

mod issue8845 {
    pub enum Something {
        Num(u8),
        TupleNums(u8, u8),
        StructNums { one: u8, two: u8 },
    }

    struct Foo(u8);

    struct Bar {
        x: u8,
        y: usize,
    }

    impl Something {
        fn get_value(&self) -> u8 {
            match self {
                Something::Num(n) => *n,
                Something::TupleNums(n, _m) => *n,
                Something::StructNums { one, two: _ } => *one,
            }
        }

        fn use_crate(&self) -> u8 {
            match self {
                crate::issue8845::Something::Num(n) => *n,
                crate::issue8845::Something::TupleNums(n, _m) => *n,
                crate::issue8845::Something::StructNums { one, two: _ } => *one,
            }
        }

        fn imported_values(&self) -> u8 {
            use Something::*;
            match self {
                Num(n) => *n,
                TupleNums(n, _m) => *n,
                StructNums { one, two: _ } => *one,
            }
        }
    }

    impl Foo {
        fn get_value(&self) -> u8 {
            let Foo(x) = self;
            *x
        }

        fn use_crate(&self) -> u8 {
            let crate::issue8845::Foo(x) = self;
            *x
        }
    }

    impl Bar {
        fn get_value(&self) -> u8 {
            let Bar { x, .. } = self;
            *x
        }

        fn use_crate(&self) -> u8 {
            let crate::issue8845::Bar { x, .. } = self;
            *x
        }
    }
}

mod issue6902 {
    use serde::Serialize;

    #[derive(Serialize)]
    pub enum Foo {
        Bar = 1,
    }
}

#[clippy::msrv = "1.36"]
fn msrv_1_36() {
    enum E {
        A,
    }

    impl E {
        fn foo(self) {
            match self {
                E::A => {},
            }
        }
    }
}

#[clippy::msrv = "1.37"]
fn msrv_1_37() {
    enum E {
        A,
    }

    impl E {
        fn foo(self) {
            match self {
                E::A => {},
            }
        }
    }
}

mod issue_10371 {
    struct Val<const V: i32> {}

    impl<const V: i32> From<Val<V>> for i32 {
        fn from(_: Val<V>) -> Self {
            todo!()
        }
    }
}
