#![feature(fn_traits, unboxed_closures)]
#![warn(clippy::no_effect_underscore_binding)]
#![allow(
    clippy::deref_addrof,
    clippy::redundant_field_names,
    clippy::uninlined_format_args,
    clippy::unnecessary_struct_initialization,
    clippy::useless_vec
)]

use std::fmt::Display;
use std::ops::{Neg, Shl};

struct Cout;

impl<T> Shl<T> for Cout
where
    T: Display,
{
    type Output = Self;
    fn shl(self, rhs: T) -> Self::Output {
        println!("{}", rhs);
        self
    }
}

impl Neg for Cout {
    type Output = Self;
    fn neg(self) -> Self::Output {
        println!("hello world");
        self
    }
}

struct Tuple(i32);
struct Struct {
    field: i32,
}
enum Enum {
    Tuple(i32),
    Struct { field: i32 },
}
struct DropStruct {
    field: i32,
}
impl Drop for DropStruct {
    fn drop(&mut self) {}
}
struct DropTuple(i32);
impl Drop for DropTuple {
    fn drop(&mut self) {}
}
enum DropEnum {
    Tuple(i32),
    Struct { field: i32 },
}
impl Drop for DropEnum {
    fn drop(&mut self) {}
}
struct FooString {
    s: String,
}
union Union {
    a: u8,
    b: f64,
}

fn get_number() -> i32 {
    0
}
fn get_struct() -> Struct {
    Struct { field: 0 }
}
fn get_drop_struct() -> DropStruct {
    DropStruct { field: 0 }
}

unsafe fn unsafe_fn() -> i32 {
    0
}

struct GreetStruct1;

impl FnOnce<(&str,)> for GreetStruct1 {
    type Output = ();

    extern "rust-call" fn call_once(self, (who,): (&str,)) -> Self::Output {
        println!("hello {}", who);
    }
}

struct GreetStruct2();

impl FnOnce<(&str,)> for GreetStruct2 {
    type Output = ();

    extern "rust-call" fn call_once(self, (who,): (&str,)) -> Self::Output {
        println!("hello {}", who);
    }
}

struct GreetStruct3;

impl FnOnce<(&str,)> for GreetStruct3 {
    type Output = ();

    extern "rust-call" fn call_once(self, (who,): (&str,)) -> Self::Output {
        println!("hello {}", who);
    }
}

fn main() {
    let s = get_struct();

    0;
    //~^ no_effect

    Tuple(0);
    //~^ no_effect

    Struct { field: 0 };
    //~^ no_effect

    Struct { ..s };
    //~^ no_effect

    Union { a: 0 };
    //~^ no_effect

    Enum::Tuple(0);
    //~^ no_effect

    Enum::Struct { field: 0 };
    //~^ no_effect

    5 + 6;
    //~^ no_effect

    *&42;
    //~^ no_effect

    &6;
    //~^ no_effect

    (5, 6, 7);
    //~^ no_effect

    ..;
    //~^ no_effect

    5..;
    //~^ no_effect

    ..5;
    //~^ no_effect

    5..6;
    //~^ no_effect

    5..=6;
    //~^ no_effect

    [42, 55];
    //~^ no_effect

    [42, 55][1];
    //~^ no_effect

    (42, 55).1;
    //~^ no_effect

    [42; 55];
    //~^ no_effect

    [42; 55][13];
    //~^ no_effect

    let mut x = 0;
    || x += 5;
    //~^ no_effect

    let s: String = "foo".into();
    FooString { s: s };
    //~^ no_effect

    let _unused = 1;
    //~^ no_effect_underscore_binding

    let _penguin = || println!("Some helpful closure");
    //~^ no_effect_underscore_binding

    let _duck = Struct { field: 0 };
    //~^ no_effect_underscore_binding

    let _cat = [2, 4, 6, 8][2];
    //~^ no_effect_underscore_binding

    let _issue_12166 = 42;
    let underscore_variable_above_can_be_used_dont_lint = _issue_12166;

    #[allow(clippy::no_effect)]
    0;

    // Do not warn
    get_number();
    unsafe { unsafe_fn() };
    let _used = get_struct();
    let _x = vec![1];
    DropStruct { field: 0 };
    DropTuple(0);
    DropEnum::Tuple(0);
    DropEnum::Struct { field: 0 };
    GreetStruct1("world");
    GreetStruct2()("world");
    GreetStruct3 {}("world");

    fn n() -> i32 {
        42
    }

    Cout << 142;
    -Cout;
}

fn issue14592() {
    struct MyStruct {
        _inner: MyInner,
    }
    struct MyInner {}

    impl Drop for MyInner {
        fn drop(&mut self) {
            println!("dropping");
        }
    }

    let x = MyStruct { _inner: MyInner {} };

    let closure = || {
        // Do not lint: dropping the assignment or assigning to `_` would
        // change the output.
        let _x = x;
    };

    println!("1");
    closure();
    println!("2");

    struct Innocuous {
        a: i32,
    }

    // Do not lint: one of the fields has a side effect.
    let x = MyInner {};
    let closure = || {
        let _x = Innocuous {
            a: {
                x;
                10
            },
        };
    };

    // Do not lint: the base has a side effect.
    let x = MyInner {};
    let closure = || {
        let _x = Innocuous {
            ..Innocuous {
                a: {
                    x;
                    10
                },
            }
        };
    };
}
