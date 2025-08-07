//! Checks the basic functionality of `std::any::type_name` for primitive types
//! and simple generic structs.

//@ run-pass

#![allow(dead_code)]

use std::any::type_name;
use std::borrow::Cow;

struct Foo<T>(T);

struct Bar<'a>(&'a u32);

struct Baz<'a, T>(&'a T);

trait TrL<'a> {}
trait TrLA<'a> {
    type A;
}
trait TrLT<'a, T> {}
trait TrLTA<'a, T> {
    type A;
}

macro_rules! t {
    ($ty:ty, $str:literal) => {
        assert_eq!(type_name::<$ty>(), $str);
    }
}

pub fn main() {
    t!(bool, "bool");
    t!(char, "char");

    t!(u8, "u8");
    t!(u16, "u16");
    t!(u32, "u32");
    t!(u64, "u64");
    t!(u128, "u128");
    t!(usize, "usize");

    t!(i8, "i8");
    t!(i16, "i16");
    t!(i32, "i32");
    t!(i64, "i64");
    t!(i128, "i128");
    t!(isize, "isize");

    t!(String, "alloc::string::String");
    t!(str, "str");
    t!(&str, "&str");
    t!(&'static str, "&str");

    t!((u16, u32, u64), "(u16, u32, u64)");
    t!([usize; 4], "[usize; 4]");
    t!([usize], "[usize]");
    t!(&[usize], "&[usize]");

    t!(*const bool, "*const bool");
    t!(*mut u64, "*mut u64");

    t!(Vec<Vec<u32>>, "alloc::vec::Vec<alloc::vec::Vec<u32>>");
    t!(Foo<usize>, "type_name_basic::Foo<usize>");
    t!(Bar<'static>, "type_name_basic::Bar");
    t!(Baz<'static, u32>, "type_name_basic::Baz<u32>");

    // FIXME: lifetime omission means these all print badly.
    t!(dyn TrL<'static>, "dyn type_name_basic::TrL<>");
    t!(dyn TrLA<'static, A = u32>, "dyn type_name_basic::TrLA<, A = u32>");
    t!(
        dyn TrLT<'static, Cow<'static, ()>>,
        "dyn type_name_basic::TrLT<, alloc::borrow::Cow<()>>"
    );
    t!(
        dyn TrLTA<'static, u32, A = Cow<'static, ()>>,
        "dyn type_name_basic::TrLTA<, u32, A = alloc::borrow::Cow<()>>"
    );

    t!(fn(i32) -> i32, "fn(i32) -> i32");
    t!(dyn for<'a> Fn(&'a u32), "dyn core::ops::function::Fn(&u32)");

    struct S<'a, T>(&'a T);
    impl<'a, T: Clone> S<'a, T> {
        fn test() {
            t!(Cow<'a, T>, "alloc::borrow::Cow<u32>");
        }
    }
    S::<u32>::test();
}
