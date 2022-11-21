// run-rustfix
// aux-build:macro_rules.rs

#![allow(unused_variables, dead_code, clippy::derive_partial_eq_without_eq)]
#![warn(clippy::equatable_if_let)]

#[macro_use]
extern crate macro_rules;

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
    if let Ordering::Greater = a.cmp(&b) {}
    if let Some(2) = c {}
    if let Struct { a: 2, b: false } = d {}
    if let Enum::TupleVariant(32, 64) = e {}
    if let Enum::RecordVariant { a: 64, b: 32 } = e {}
    if let Enum::UnitVariant = e {}
    if let (Enum::UnitVariant, &Struct { a: 2, b: false }) = (e, &d) {}

    // false

    if let 2 | 3 = a {}
    if let x @ 2 = a {}
    if let Some(3 | 4) = c {}
    if let Struct { a, b: false } = d {}
    if let Struct { a: 2, b: x } = d {}
    if let NotPartialEq::A = f {}
    if let NotStructuralEq::A = g {}
    if let Some(NotPartialEq::A) = Some(f) {}
    if let Some(NotStructuralEq::A) = Some(g) {}
    if let NoPartialEqStruct { a: 2, b: false } = h {}

    macro_rules! m1 {
        (x) => {
            "abc"
        };
    }
    if let m1!(x) = "abc" {
        println!("OK");
    }

    equatable_if_let!(a);
}
