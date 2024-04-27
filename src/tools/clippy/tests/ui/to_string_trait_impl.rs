#![warn(clippy::to_string_trait_impl)]
#![feature(min_specialization)]

use std::fmt::{self, Display};

struct Point {
    x: usize,
    y: usize,
}

impl ToString for Point {
    fn to_string(&self) -> String {
        format!("({}, {})", self.x, self.y)
    }
}

struct Foo;

impl Display for Foo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Foo")
    }
}

struct Bar;

impl Bar {
    #[allow(clippy::inherent_to_string)]
    fn to_string(&self) -> String {
        String::from("Bar")
    }
}

mod issue12263 {
    pub struct MyStringWrapper<'a>(&'a str);

    impl std::fmt::Display for MyStringWrapper<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.0.fmt(f)
        }
    }

    impl ToString for MyStringWrapper<'_> {
        fn to_string(&self) -> String {
            self.0.to_string()
        }
    }

    pub struct S<T>(T);
    impl std::fmt::Display for S<String> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            todo!()
        }
    }
    // no specialization if the generics differ, so lint
    impl ToString for S<i32> {
        fn to_string(&self) -> String {
            todo!()
        }
    }

    pub struct S2<T>(T);
    impl std::fmt::Display for S2<String> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            todo!()
        }
    }

    // also specialization if the generics don't differ
    impl ToString for S2<String> {
        fn to_string(&self) -> String {
            todo!()
        }
    }
}
