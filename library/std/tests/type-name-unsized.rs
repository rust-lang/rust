#![allow(dead_code)]

use std::fmt::Debug;

struct NT(str);

struct DST {
    a: u32,
    b: str,
}

macro_rules! check {
    (val: $ty_of:expr, $expected:expr) => {
        assert_eq!(type_name_of_val($ty_of), $expected);
    };
    ($ty:ty, $expected:expr) => {
        assert_eq!(std::any::type_name::<$ty>(), $expected);
    };
}

/// Tests that [`std::any::type_name`] supports unsized types.
#[test]
fn type_name_unsized() {
    check!([u8], "[u8]");
    check!(str, "str");
    check!(dyn Send, "dyn core::marker::Send");
    check!(NT, "type_name_unsized::NT");
    check!(DST, "type_name_unsized::DST");
    check!(&i32, "&i32");
    check!(&'static i32, "&i32");
    check!((i32, u32), "(i32, u32)");
    check!(val: foo(), "type_name_unsized::Foo");
    check!(val: Foo::new, "type_name_unsized::Foo::new");
    check!(val:
        <Foo as Debug>::fmt,
        "<type_name_unsized::Foo as core::fmt::Debug>::fmt"
    );
    check!(val: || {}, "type_name_unsized::type_name_unsized::{{closure}}");
    bar::<i32>();
}

trait Trait {
    type Assoc;
}

impl Trait for i32 {
    type Assoc = String;
}

fn bar<T: Trait>() {
    check!(T::Assoc, "alloc::string::String");
    check!(T, "i32");
}

fn type_name_of_val<T>(_: T) -> &'static str {
    std::any::type_name::<T>()
}

#[derive(Debug)]
struct Foo;

impl Foo {
    fn new() -> Self {
        Foo
    }
}

fn foo() -> impl Debug {
    Foo
}
