#![feature(exhaustive_patterns, never_type)]
#![allow(dead_code, unreachable_code, unused_variables)]
#![allow(clippy::let_and_return, clippy::uninhabited_references)]

enum SingleVariantEnum {
    Variant(i32),
}

struct TupleStruct(i32);

struct NonCopy;
struct TupleStructWithNonCopy(NonCopy);

enum EmptyEnum {}

macro_rules! match_enum {
    ($param:expr) => {
        let data = match $param {
            SingleVariantEnum::Variant(i) => i,
        };
    };
}

fn infallible_destructuring_match_enum() {
    let wrapper = SingleVariantEnum::Variant(0);

    // This should lint!
    let data = match wrapper {
        //~^ infallible_destructuring_match
        SingleVariantEnum::Variant(i) => i,
    };

    // This shouldn't (inside macro)
    match_enum!(wrapper);

    // This shouldn't!
    let data = match wrapper {
        SingleVariantEnum::Variant(_) => -1,
    };

    // Neither should this!
    let data = match wrapper {
        SingleVariantEnum::Variant(i) => -1,
    };

    let SingleVariantEnum::Variant(data) = wrapper;
}

macro_rules! match_struct {
    ($param:expr) => {
        let data = match $param {
            TupleStruct(i) => i,
        };
    };
}

fn infallible_destructuring_match_struct() {
    let wrapper = TupleStruct(0);

    // This should lint!
    let data = match wrapper {
        //~^ infallible_destructuring_match
        TupleStruct(i) => i,
    };

    // This shouldn't (inside macro)
    match_struct!(wrapper);

    // This shouldn't!
    let data = match wrapper {
        TupleStruct(_) => -1,
    };

    // Neither should this!
    let data = match wrapper {
        TupleStruct(i) => -1,
    };

    let TupleStruct(data) = wrapper;
}

fn infallible_destructuring_match_struct_with_noncopy() {
    let wrapper = TupleStructWithNonCopy(NonCopy);

    // This should lint! (keeping `ref` in the suggestion)
    let data = match wrapper {
        //~^ infallible_destructuring_match
        TupleStructWithNonCopy(ref n) => n,
    };

    let TupleStructWithNonCopy(ref data) = wrapper;
}

macro_rules! match_never_enum {
    ($param:expr) => {
        let data = match $param {
            Ok(i) => i,
        };
    };
}

fn never_enum() {
    let wrapper: Result<i32, !> = Ok(23);

    // This should lint!
    let data = match wrapper {
        //~^ infallible_destructuring_match
        Ok(i) => i,
    };

    // This shouldn't (inside macro)
    match_never_enum!(wrapper);

    // This shouldn't!
    let data = match wrapper {
        Ok(_) => -1,
    };

    // Neither should this!
    let data = match wrapper {
        Ok(i) => -1,
    };

    let Ok(data) = wrapper;
}

impl EmptyEnum {
    fn match_on(&self) -> ! {
        // The lint shouldn't pick this up, as `let` won't work here!
        let data = match *self {};
        data
    }
}

fn main() {}
