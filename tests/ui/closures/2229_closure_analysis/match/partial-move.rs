// This test measures the effect of matching-induced partial captures on the borrow checker.
// In particular, in each of the cases below, the closure either captures the entire enum/struct,
// or each field separately.
//
// If the entire ADT gets captured, it'll happen by move, and the closure will live for 'static.
// On the other hand, if each field gets captured separately, the u32 field, being Copy, will only
// get captured by an immutable borrow, resulting in a borrow checker error.
//
// See rust-lang/rust#147722
//
//@ edition:2021
//@ aux-build:partial_move_lib.rs
pub struct Struct(u32, String);

pub enum Enum {
    A(u32, String),
}

pub enum TwoVariants {
    A(u32, String),
    B,
}

#[non_exhaustive]
pub enum NonExhaustive {
    A(u32, String),
}

extern crate partial_move_lib;
use partial_move_lib::ExtNonExhaustive;

// First, let's assert that the additional wildcard arm is not a source of any behavior
// differences:
pub fn test_enum1(x: Enum) -> impl FnOnce() {
    || {
    //~^ ERROR: closure may outlive the current function, but it borrows `x.0`
        match x {
            Enum::A(a, b) => {
                drop((a, b));
            }
            _ => unreachable!(),
        }
    }
}

pub fn test_enum2(x: Enum) -> impl FnOnce() {
    || {
    //~^ ERROR: closure may outlive the current function, but it borrows `x.0`
        match x {
            Enum::A(a, b) => {
                drop((a, b));
            }
        }
    }
}

// The behavior for single-variant enums matches what happens for a struct
pub fn test_struct(x: Struct) -> impl FnOnce() {
    || {
    //~^ ERROR: closure may outlive the current function, but it borrows `x.0`
        match x {
            Struct(a, b) => {
                drop((a, b));
            }
        }
    }
}

// If we have two variants, the entire enum gets moved into the closure
pub fn test_two_variants(x: TwoVariants) -> impl FnOnce() {
    || {
        match x {
            TwoVariants::A(a, b) => {
                drop((a, b));
            }
            _ => unreachable!(),
        }
    }
}

// ...and single-variant, non-exhaustive enums *should* behave as if they had multiple variants
pub fn test_non_exhaustive1(x: NonExhaustive) -> impl FnOnce() {
    || {
    //~^ ERROR: closure may outlive the current function, but it borrows `x.0`
        match x {
            NonExhaustive::A(a, b) => {
                drop((a, b));
            }
            _ => unreachable!(),
        }
    }
}

// (again, wildcard branch or not)
pub fn test_non_exhaustive2(x: NonExhaustive) -> impl FnOnce() {
    || {
    //~^ ERROR: closure may outlive the current function, but it borrows `x.0`
        match x {
            NonExhaustive::A(a, b) => {
                drop((a, b));
            }
        }
    }
}

// ...regardless of whether the enum is defined in the current, or in another crate
pub fn test_ext(x: ExtNonExhaustive) -> impl FnOnce() {
    || {
        match x {
            ExtNonExhaustive::A(a, b) => {
                drop((a, b));
            }
            _ => unreachable!(),
        }
    }
}

fn main() {}
