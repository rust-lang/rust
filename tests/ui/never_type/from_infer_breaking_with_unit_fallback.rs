// issue: rust-lang/rust#66757
//
// This is a *minimization* of the issue.
// Note that the original version with the `?` does not fail anymore even with fallback to unit,
// see `tests/ui/never_type/question_mark_from_never.rs`.
//
//@ revisions: unit never
//@[never] check-pass
#![allow(internal_features)]
#![feature(rustc_attrs, never_type)]
#![cfg_attr(unit, rustc_never_type_options(fallback = "unit"))]
#![cfg_attr(never, rustc_never_type_options(fallback = "never"))]

struct E;

impl From<!> for E {
    fn from(_: !) -> E {
        E
    }
}

#[allow(unreachable_code)]
fn foo(never: !) {
    <E as From<!>>::from(never); // Ok
    <E as From<_>>::from(never); // Should the inference fail?
    //[unit]~^ error: the trait bound `E: From<()>` is not satisfied
}

fn main() {}
