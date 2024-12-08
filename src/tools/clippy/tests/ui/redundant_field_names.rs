//@aux-build:proc_macros.rs
#![warn(clippy::redundant_field_names)]
#![allow(clippy::extra_unused_type_parameters, clippy::no_effect, dead_code, unused_variables)]

#[macro_use]
extern crate proc_macros;

use std::ops::{Range, RangeFrom, RangeInclusive, RangeTo, RangeToInclusive};

mod foo {
    pub const BAR: u8 = 0;
}

struct Person {
    gender: u8,
    age: u8,
    name: u8,
    buzz: u64,
    foo: u8,
}

pub struct S {
    v: usize,
}

fn main() {
    let gender: u8 = 42;
    let age = 0;
    let fizz: u64 = 0;
    let name: u8 = 0;

    let me = Person {
        gender: gender,
        age: age,

        name,          //should be ok
        buzz: fizz,    //should be ok
        foo: foo::BAR, //should be ok
    };

    // Range expressions
    let (start, end) = (0, 0);

    let _ = start..;
    let _ = ..end;
    let _ = start..end;

    let _ = ..=end;
    let _ = start..=end;

    // Issue #2799
    let _: Vec<_> = (start..end).collect();

    // hand-written Range family structs are linted
    let _ = RangeFrom { start: start };
    let _ = RangeTo { end: end };
    let _ = Range { start: start, end: end };
    let _ = RangeInclusive::new(start, end);
    let _ = RangeToInclusive { end: end };

    external! {
        let v = 1;
        let _ = S {
            v: v
        };
    }

    let v = 2;
    macro_rules! internal {
        ($i:ident) => {
            let _ = S { v: v };
            let _ = S { $i: v };
            let _ = S { v: $i };
            let _ = S { $i: $i };
        };
    }
    internal!(v);
}

fn issue_3476() {
    fn foo<T>() {}

    struct S {
        foo: fn(),
    }

    S { foo: foo::<i32> };
}

#[clippy::msrv = "1.16"]
fn msrv_1_16() {
    let start = 0;
    let _ = RangeFrom { start: start };
}

#[clippy::msrv = "1.17"]
fn msrv_1_17() {
    let start = 0;
    let _ = RangeFrom { start: start };
}
