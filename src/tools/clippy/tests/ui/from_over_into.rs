#![feature(type_alias_impl_trait)]
#![warn(clippy::from_over_into)]
#![allow(non_local_definitions)]
#![allow(unused)]

// this should throw an error
struct StringWrapper(String);

impl Into<StringWrapper> for String {
    //~^ from_over_into
    fn into(self) -> StringWrapper {
        StringWrapper(self)
    }
}

struct SelfType(String);

impl Into<SelfType> for String {
    //~^ from_over_into
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
    //~^ from_over_into
    fn into(self) -> SelfKeywords {
        let _ = Self;
        let _ = Self::FOO;
        let _: Self = self;

        SelfKeywords
    }
}

struct ExplicitPaths(bool);

impl core::convert::Into<bool> for crate::ExplicitPaths {
    //~^ from_over_into
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

struct PathInExpansion;

impl Into<String> for PathInExpansion {
    //~^ from_over_into
    fn into(self) -> String {
        // non self/Self paths in expansions are fine
        panic!()
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
        //~^ from_over_into
        fn into(self) -> FromOverInto<T> {
            FromOverInto(self)
        }
    }
}

fn issue_12138() {
    struct Hello;

    impl Into<()> for Hello {
        //~^ from_over_into
        fn into(self) {}
    }
}

fn issue_112502() {
    struct MyInt(i64);

    impl Into<i64> for MyInt {
        //~^ from_over_into
        fn into(self: MyInt) -> i64 {
            self.0
        }
    }
}

fn main() {}
