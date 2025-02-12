//@no-rustfix

#![warn(clippy::unconditional_recursion)]
#![allow(
    clippy::partialeq_ne_impl,
    clippy::default_constructed_unit_structs,
    clippy::only_used_in_recursion,
    clippy::needless_lifetimes
)]

enum Foo {
    A,
    B,
}

impl PartialEq for Foo {
    fn ne(&self, other: &Self) -> bool {
        //~^ unconditional_recursion

        self != other
    }
    fn eq(&self, other: &Self) -> bool {
        //~^ unconditional_recursion

        self == other
    }
}

enum Foo2 {
    A,
    B,
}

impl PartialEq for Foo2 {
    fn ne(&self, other: &Self) -> bool {
        //~^ unconditional_recursion
        self != &Foo2::B // no error here
    }
    fn eq(&self, other: &Self) -> bool {
        //~^ unconditional_recursion
        self == &Foo2::B // no error here
    }
}

enum Foo3 {
    A,
    B,
}

impl PartialEq for Foo3 {
    fn ne(&self, other: &Self) -> bool {
        //~^ unconditional_recursion
        //~| ERROR: function cannot return without recursing
        self.ne(other)
    }
    fn eq(&self, other: &Self) -> bool {
        //~^ unconditional_recursion
        //~| ERROR: function cannot return without recursing

        self.eq(other)
    }
}

enum Foo4 {
    A,
    B,
}

impl PartialEq for Foo4 {
    fn ne(&self, other: &Self) -> bool {
        self.eq(other) // no error
    }
    fn eq(&self, other: &Self) -> bool {
        self.ne(other) // no error
    }
}

enum Foo5 {
    A,
    B,
}

impl Foo5 {
    fn a(&self) -> bool {
        true
    }
}

impl PartialEq for Foo5 {
    fn ne(&self, other: &Self) -> bool {
        self.a() // no error
    }
    fn eq(&self, other: &Self) -> bool {
        self.a() // no error
    }
}

struct S;

// Check the order doesn't matter.
impl PartialEq for S {
    fn ne(&self, other: &Self) -> bool {
        //~^ unconditional_recursion

        other != self
    }
    fn eq(&self, other: &Self) -> bool {
        //~^ unconditional_recursion

        other == self
    }
}

struct S2;

// Check that if the same element is compared, it's also triggering the lint.
impl PartialEq for S2 {
    fn ne(&self, other: &Self) -> bool {
        //~^ unconditional_recursion

        other != other
        //~^ eq_op
    }
    fn eq(&self, other: &Self) -> bool {
        //~^ unconditional_recursion

        other == other
        //~^ eq_op
    }
}

struct S3;

impl PartialEq for S3 {
    fn ne(&self, _other: &Self) -> bool {
        //~^ unconditional_recursion

        self != self
        //~^ eq_op
    }
    fn eq(&self, _other: &Self) -> bool {
        //~^ unconditional_recursion

        self == self
        //~^ eq_op
    }
}

// There should be no warning here!
#[derive(PartialEq)]
enum E {
    A,
    B,
}

#[derive(PartialEq)]
struct Bar<T: PartialEq>(T);

struct S4;

impl PartialEq for S4 {
    fn eq(&self, other: &Self) -> bool {
        // No warning here.
        Bar(self) == Bar(other)
    }
}

macro_rules! impl_partial_eq {
    ($ty:ident) => {
        impl PartialEq for $ty {
            fn eq(&self, other: &Self) -> bool {
                //~^ unconditional_recursion

                self == other
            }
        }
    };
}

struct S5;

impl_partial_eq!(S5);

struct S6 {
    field: String,
}

impl PartialEq for S6 {
    fn eq(&self, other: &Self) -> bool {
        let mine = &self.field;
        let theirs = &other.field;
        mine == theirs // Should not warn!
    }
}

struct S7<'a> {
    field: &'a S7<'a>,
}

impl<'a> PartialEq for S7<'a> {
    fn eq(&self, other: &Self) -> bool {
        //~^ unconditional_recursion

        let mine = &self.field;
        let theirs = &other.field;
        mine == theirs
    }
}

struct S8 {
    num: i32,
    field: Option<Box<S8>>,
}

impl PartialEq for S8 {
    fn eq(&self, other: &Self) -> bool {
        if self.num != other.num {
            return false;
        }

        let (this, other) = match (self.field.as_deref(), other.field.as_deref()) {
            (Some(x1), Some(x2)) => (x1, x2),
            (None, None) => return true,
            _ => return false,
        };

        this == other
    }
}

struct S9;

#[allow(clippy::to_string_trait_impl)]
impl std::string::ToString for S9 {
    fn to_string(&self) -> String {
        //~^ ERROR: function cannot return without recursing
        self.to_string()
    }
}

struct S10;

#[allow(clippy::to_string_trait_impl)]
impl std::string::ToString for S10 {
    fn to_string(&self) -> String {
        //~^ ERROR: function cannot return without recursing
        let x = self;
        x.to_string()
    }
}

struct S11;

#[allow(clippy::to_string_trait_impl)]
impl std::string::ToString for S11 {
    fn to_string(&self) -> String {
        //~^ ERROR: function cannot return without recursing
        (self as &Self).to_string()
    }
}

struct S12;

impl std::default::Default for S12 {
    fn default() -> Self {
        Self::new()
    }
}

impl S12 {
    fn new() -> Self {
        //~^ unconditional_recursion

        Self::default()
    }

    fn bar() -> Self {
        // Should not warn!
        Self::default()
    }
}

#[derive(Default)]
struct S13 {
    f: u32,
}

impl S13 {
    fn new() -> Self {
        // Should not warn!
        Self::default()
    }
}

struct S14 {
    field: String,
}

impl PartialEq for S14 {
    fn eq(&self, other: &Self) -> bool {
        // Should not warn!
        self.field.eq(&other.field)
    }
}

struct S15<'a> {
    field: &'a S15<'a>,
}

impl PartialEq for S15<'_> {
    fn eq(&self, other: &Self) -> bool {
        //~^ unconditional_recursion

        let mine = &self.field;
        let theirs = &other.field;
        mine.eq(theirs)
    }
}

mod issue12154 {
    struct MyBox<T>(T);

    impl<T> std::ops::Deref for MyBox<T> {
        type Target = T;
        fn deref(&self) -> &T {
            &self.0
        }
    }

    impl<T: PartialEq> PartialEq for MyBox<T> {
        fn eq(&self, other: &Self) -> bool {
            (**self).eq(&**other)
        }
    }

    // Not necessarily related to the issue but another FP from the http crate that was fixed with it:
    // https://docs.rs/http/latest/src/http/header/name.rs.html#1424
    // We used to simply peel refs from the LHS and RHS, so we couldn't differentiate
    // between `PartialEq<T> for &T` and `PartialEq<&T> for T` impls.
    #[derive(PartialEq)]
    struct HeaderName;
    impl<'a> PartialEq<&'a HeaderName> for HeaderName {
        fn eq(&self, other: &&'a HeaderName) -> bool {
            *self == **other
        }
    }

    impl<'a> PartialEq<HeaderName> for &'a HeaderName {
        fn eq(&self, other: &HeaderName) -> bool {
            *other == *self
        }
    }

    // Issue #12181 but also fixed by the same PR
    struct Foo;

    impl Foo {
        fn as_str(&self) -> &str {
            "Foo"
        }
    }

    impl PartialEq for Foo {
        fn eq(&self, other: &Self) -> bool {
            self.as_str().eq(other.as_str())
        }
    }

    impl<T> PartialEq<T> for Foo
    where
        for<'a> &'a str: PartialEq<T>,
    {
        fn eq(&self, other: &T) -> bool {
            (&self.as_str()).eq(other)
        }
    }
}

// From::from -> Into::into -> From::from
struct BadFromTy1<'a>(&'a ());
struct BadIntoTy1<'b>(&'b ());
impl<'a> From<BadFromTy1<'a>> for BadIntoTy1<'static> {
    fn from(f: BadFromTy1<'a>) -> Self {
        //~^ unconditional_recursion
        f.into()
    }
}

// Using UFCS syntax
struct BadFromTy2<'a>(&'a ());
struct BadIntoTy2<'b>(&'b ());
impl<'a> From<BadFromTy2<'a>> for BadIntoTy2<'static> {
    fn from(f: BadFromTy2<'a>) -> Self {
        //~^ unconditional_recursion
        Into::into(f)
    }
}

// Different Into impl (<i16 as Into<i32>>), so no infinite recursion
struct BadFromTy3;
impl From<BadFromTy3> for i32 {
    fn from(f: BadFromTy3) -> Self {
        Into::into(1i16)
    }
}

// A conditional return that ends the recursion
struct BadFromTy4;
impl From<BadFromTy4> for i32 {
    fn from(f: BadFromTy4) -> Self {
        if true {
            return 42;
        }
        f.into()
    }
}

// Types differ in refs, don't lint
impl From<&BadFromTy4> for i32 {
    fn from(f: &BadFromTy4) -> Self {
        BadFromTy4.into()
    }
}

fn main() {}
