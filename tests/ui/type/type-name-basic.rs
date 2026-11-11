//! Checks the basic functionality of `std::any::type_name` for primitive types
//! and simple generic structs.

//@ run-pass

#![allow(dead_code)]

use std::any::{Any, type_name, type_name_of_val};
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

macro_rules! v {
    ($v:expr, $str:literal) => {
        assert_eq!(type_name_of_val(&$v), $str);
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
    t!(Bar<'static>, "type_name_basic::Bar<'_>");
    t!(Baz<'static, u32>, "type_name_basic::Baz<'_, u32>");

    t!(dyn TrL<'static>, "dyn type_name_basic::TrL<'_>");
    t!(dyn TrLA<'static, A = u32>, "dyn type_name_basic::TrLA<'_, A = u32>");
    t!(
        dyn TrLT<'static, Cow<'static, ()>>,
        "dyn type_name_basic::TrLT<'_, alloc::borrow::Cow<'_, ()>>"
    );
    t!(
        dyn TrLTA<'static, u32, A = Cow<'static, ()>>,
        "dyn type_name_basic::TrLTA<'_, u32, A = alloc::borrow::Cow<'_, ()>>"
    );

    t!(fn(i32) -> i32, "fn(i32) -> i32");
    t!(fn(&'static u32), "fn(&u32)");

    // FIXME: these are sub-optimal, ideally the `for<...>` would be printed.
    t!(for<'a> fn(&'a u32), "fn(&'_ u32)");
    t!(for<'a, 'b> fn(&'a u32, &'b u32), "fn(&'_ u32, &'_ u32)");
    t!(for<'a> fn(for<'b> fn(&'a u32, &'b u32)), "fn(fn(&'_ u32, &'_ u32))");

    struct S<'a, T>(&'a T);
    impl<'a, T: Clone> S<'a, T> {
        fn test() {
            t!(Cow<'a, T>, "alloc::borrow::Cow<'_, u32>");
        }
    }
    S::<u32>::test();

    struct Wrap<T>(T);
    impl Wrap<&()> {
        fn get(&self) -> impl Any {
            struct Info;
            Info
        }
    }
    let a = Wrap(&()).get();
    v!(a, "type_name_basic::main::Wrap<&()>::get::Info");

    struct Issue146249<T>(T);
    impl Issue146249<Box<dyn FnOnce()>> {
        pub fn bar(&self) {
            let f = || {};
            v!(
                f,
                "type_name_basic::main::Issue146249<\
                    alloc::boxed::Box<dyn core::ops::function::FnOnce()>\
                >::bar::{{closure}}"
            );
        }
    }
    let v: Issue146249<Box<dyn FnOnce()>> = Issue146249(Box::new(|| {}));
    v.bar();
}
