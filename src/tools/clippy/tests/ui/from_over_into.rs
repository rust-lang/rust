//@run-rustfix

#![feature(type_alias_impl_trait)]
#![warn(clippy::from_over_into)]
#![allow(unused)]

// this should throw an error
struct StringWrapper(String);

impl Into<StringWrapper> for String {
    fn into(self) -> StringWrapper {
        StringWrapper(self)
    }
}

struct SelfType(String);

impl Into<SelfType> for String {
    fn into(self) -> SelfType {
        SelfType(Self::new())
    }
}

#[derive(Default)]
struct X;

impl X {
    const FOO: &'static str = "a";
}

struct SelfKeywords;

impl Into<SelfKeywords> for X {
    fn into(self) -> SelfKeywords {
        let _ = Self;
        let _ = Self::FOO;
        let _: Self = self;

        SelfKeywords
    }
}

struct ExplicitPaths(bool);

impl core::convert::Into<bool> for crate::ExplicitPaths {
    fn into(mut self) -> bool {
        let in_closure = || self.0;

        self.0 = false;
        self.0
    }
}

// this is fine
struct A(String);

impl From<String> for A {
    fn from(s: String) -> A {
        A(s)
    }
}

#[clippy::msrv = "1.40"]
fn msrv_1_40() {
    struct FromOverInto<T>(Vec<T>);

    impl<T> Into<FromOverInto<T>> for Vec<T> {
        fn into(self) -> FromOverInto<T> {
            FromOverInto(self)
        }
    }
}

#[clippy::msrv = "1.41"]
fn msrv_1_41() {
    struct FromOverInto<T>(Vec<T>);

    impl<T> Into<FromOverInto<T>> for Vec<T> {
        fn into(self) -> FromOverInto<T> {
            FromOverInto(self)
        }
    }
}

fn main() {}
