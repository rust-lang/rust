//@aux-build:proc_macros.rs

#![allow(
    unused_variables,
    dead_code,
    clippy::derive_partial_eq_without_eq,
    clippy::needless_if
)]
#![warn(clippy::equatable_if_let)]

extern crate proc_macros;
use proc_macros::{external, inline_macros};

use std::cmp::Ordering;

#[derive(PartialEq)]
enum Enum {
    TupleVariant(i32, u64),
    RecordVariant { a: i64, b: u32 },
    UnitVariant,
    Recursive(Struct),
}

#[derive(PartialEq)]
struct Struct {
    a: i32,
    b: bool,
}

struct NoPartialEqStruct {
    a: i32,
    b: bool,
}

enum NotPartialEq {
    A,
    B,
}

enum NotStructuralEq {
    A,
    B,
}

impl PartialEq for NotStructuralEq {
    fn eq(&self, _: &NotStructuralEq) -> bool {
        false
    }
}

#[inline_macros]
fn main() {
    let a = 2;
    let b = 3;
    let c = Some(2);
    let d = Struct { a: 2, b: false };
    let e = Enum::UnitVariant;
    let f = NotPartialEq::A;
    let g = NotStructuralEq::A;
    let h = NoPartialEqStruct { a: 2, b: false };

    // true

    if let 2 = a {}
    //~^ equatable_if_let
    if let Ordering::Greater = a.cmp(&b) {}
    //~^ equatable_if_let
    if let Some(2) = c {}
    //~^ equatable_if_let
    if let Struct { a: 2, b: false } = d {}
    //~^ equatable_if_let
    if let Enum::TupleVariant(32, 64) = e {}
    //~^ equatable_if_let
    if let Enum::RecordVariant { a: 64, b: 32 } = e {}
    //~^ equatable_if_let
    if let Enum::UnitVariant = e {}
    //~^ equatable_if_let
    if let (Enum::UnitVariant, &Struct { a: 2, b: false }) = (e, &d) {}
    //~^ equatable_if_let

    // false

    if let 2 | 3 = a {}
    if let x @ 2 = a {}
    if let Some(3 | 4) = c {}
    if let Struct { a, b: false } = d {}
    if let Struct { a: 2, b: x } = d {}
    if let NotPartialEq::A = f {}
    //~^ equatable_if_let
    if let NotStructuralEq::A = g {}
    //~^ equatable_if_let
    if let Some(NotPartialEq::A) = Some(f) {}
    //~^ equatable_if_let
    if let Some(NotStructuralEq::A) = Some(g) {}
    //~^ equatable_if_let
    if let NoPartialEqStruct { a: 2, b: false } = h {}
    //~^ equatable_if_let

    if let inline!("abc") = "abc" {
        //~^ equatable_if_let
        println!("OK");
    }

    external!({ if let 2 = $a {} });
}

mod issue8710 {
    fn str_ref(cs: &[char]) {
        if let Some('i') = cs.iter().next() {
            //~^ equatable_if_let
        } else {
            todo!();
        }
    }

    fn i32_ref(cs: &[i32]) {
        if let Some(1) = cs.iter().next() {
            //~^ equatable_if_let
        } else {
            todo!();
        }
    }

    fn enum_ref() {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
        enum MyEnum {
            A(i32),
            B,
        }

        fn get_enum() -> Option<&'static MyEnum> {
            todo!()
        }

        if let Some(MyEnum::B) = get_enum() {
            //~^ equatable_if_let
        } else {
            todo!();
        }
    }
}
