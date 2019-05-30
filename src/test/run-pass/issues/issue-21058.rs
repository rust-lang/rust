// run-pass
#![allow(dead_code)]
#![feature(core_intrinsics)]

use std::fmt::Debug;

struct NT(str);
struct DST { a: u32, b: str }

macro_rules! check {
    (val: $ty_of:expr, $expected:expr) => {
        assert_eq!(type_name_of_val($ty_of), $expected);
    };
    ($ty:ty, $expected:expr) => {
        assert_eq!(unsafe { std::intrinsics::type_name::<$ty>()}, $expected);
    };
}

fn main() {
    // type_name should support unsized types
    check!([u8], "[u8]");
    check!(str, "str");
    check!(dyn Send, "dyn core::marker::Send");
    check!(NT, "issue_21058::NT");
    check!(DST, "issue_21058::DST");
    check!(&i32, "&i32");
    check!(&'static i32, "&i32");
    check!((i32, u32), "(i32, u32)");
    check!(val: foo(), "issue_21058::Foo");
    check!(val: Foo::new, "issue_21058::Foo::new");
    check!(val:
        <Foo as Debug>::fmt,
        "<issue_21058::Foo as core::fmt::Debug>::fmt"
    );
    check!(val: || {}, "issue_21058::main::{{closure}}");
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
    unsafe { std::intrinsics::type_name::<T>() }
}

#[derive(Debug)]
struct Foo;

impl Foo {
    fn new() -> Self { Foo }
}

fn foo() -> impl Debug {
    Foo
}
