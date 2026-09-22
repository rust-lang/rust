//! Regression test for https://github.com/rust-lang/rust/issues/131327.
//! A runtime range endpoint is invalid regardless of its type.

//@ revisions: generic chars both inferred wrong_type late_inferred
#![allow(dead_code, unreachable_patterns, unused_variables)]

#[cfg(generic)]
fn cmp<T: PartialOrd, R>(x: T, y: T, smaller: R, equal: R, greater: R) -> R {
    match x {
        ..y => smaller,
        //[generic]~^ ERROR runtime values cannot be referenced in patterns
        y => equal,
        _ => greater,
    }
}

#[cfg(chars)]
fn chars(x: char, y: char) {
    match x {
        ..y => (),
        //[chars]~^ ERROR runtime values cannot be referenced in patterns
        _ => (),
    }
}

#[cfg(both)]
fn both<T: PartialOrd>(x: T, start: T, end: T) {
    match x {
        start..=end => (),
        //[both]~^ ERROR runtime values cannot be referenced in patterns
        // Both endpoints are labeled by the same diagnostic.
        _ => (),
    }
}

#[cfg(inferred)]
fn inferred() {
    let end = Default::default();
    match 0 {
        ..end => (),
        //[inferred]~^ ERROR runtime values cannot be referenced in patterns
        _ => (),
    }
}

const LOWER: u8 = 1;
const UPPER: u8 = 9;
fn constants(x: u8) -> bool {
    matches!(x, LOWER..=UPPER)
}

#[cfg(wrong_type)]
fn wrong_type(x: bool) {
    match x {
        false..=true => (),
        //[wrong_type]~^ ERROR only `char` and numeric types are allowed in range patterns
        _ => (),
    }
}

#[cfg(late_inferred)]
fn late_inferred() {
    #[derive(Default)]
    struct Bound;

    let end = Default::default();
    match Bound {
        ..end => (),
        //[late_inferred]~^ ERROR runtime values cannot be referenced in patterns
        _ => (),
    }
}

fn main() {}
