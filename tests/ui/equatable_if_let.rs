//@aux-build:proc_macros.rs
#![allow(clippy::derive_partial_eq_without_eq, clippy::needless_ifs)]
#![warn(clippy::equatable_if_let)]
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

#[inline_macros]
fn issue14548() {
    if let inline!("abc") = "abc" {
        println!("OK");
    }

    let a = 2;
    external!({ if let 2 = $a {} });

    // Don't lint: `==`/`matches!` might be correct for a particular `$($font)|*`, but not in general
    macro_rules! m1 {
        ($($font:pat_param)|*) => {
            if let $($font)|* = "from_expansion" {}
        }
    }
    m1!("foo");
    m1!("Sans" | "Serif" | "Sans Mono");
    m1!(inline!("foo"));

    // Don't lint: the suggestion might be correct for a particular `$from_root_ctxt`, but not in
    // general
    macro_rules! m2 {
        ($from_root_ctxt:pat) => {
            if let $from_root_ctxt = "from_expansion" {}
        };
    }
    m2!("foo");
    m2!("Sans" | "Serif" | "Sans Mono");
    m2!(inline!("foo"));

    // Don't lint: the suggestion might be correct for a particular `$from_root_ctxt`, but not in
    // general
    macro_rules! m3 {
        ($from_root_ctxt:expr) => {
            if let "from_expansion" = $from_root_ctxt {}
        };
    }
    m3!("foo");
    m3!("foo");
    m3!(inline!("foo"));

    // Don't lint: the suggestion might be correct for a particular `$from_root_ctxt`, but not in
    // general. Don't get confused by the scrutinee coming from macro invocation
    macro_rules! m4 {
        ($from_root_ctxt:pat) => {
            if let $from_root_ctxt = inline!("from_expansion") {}
        };
    }
    m4!("foo");
    m4!("Sans" | "Serif" | "Sans Mono");
    m4!(inline!("foo"));

    // Don't lint: the suggestion might be correct for a particular `$from_root_ctxt`, but not in
    // general. Don't get confused by the scrutinee coming from macro invocation
    macro_rules! m5 {
        ($from_root_ctxt:expr) => {
            if let inline!("from_expansion") = $from_root_ctxt {}
        };
    }
    m5!("foo");
    m5!("foo");
    m5!(inline!("foo"));

    // Would be nice to lint: both sides are macro _invocations_, so the suggestion is correct in
    // general
    if let inline!("foo") = inline!("bar") {}
}

// PartialEq is not stable in consts yet
fn issue15376() {
    enum NonConstEq {
        A,
        B,
    }
    impl PartialEq for NonConstEq {
        fn eq(&self, _other: &Self) -> bool {
            true
        }
    }

    const N: NonConstEq = NonConstEq::A;

    // `impl PartialEq` is not const, suggest `matches!`
    const _: u32 = if let NonConstEq::A = N { 0 } else { 1 };
    //~^ ERROR: this pattern matching can be expressed using `matches!`
    const _: u32 = if let Some(NonConstEq::A) = Some(N) { 0 } else { 1 };
    //~^ ERROR: this pattern matching can be expressed using `matches!`
}
