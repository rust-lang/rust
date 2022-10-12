// aux-build:empty.rs
// revisions: normal exhaustive_patterns
//
// This tests a match with no arms on various types.
#![feature(never_type)]
#![feature(never_type_fallback)]
#![cfg_attr(exhaustive_patterns, feature(exhaustive_patterns))]
#![deny(unreachable_patterns)]

extern crate empty;

enum EmptyEnum {}

struct NonEmptyStruct1;
struct NonEmptyStruct2(bool);
union NonEmptyUnion1 {
    foo: (),
}
union NonEmptyUnion2 {
    foo: (),
    bar: (),
}
enum NonEmptyEnum1 {
    Foo(bool),
}
enum NonEmptyEnum2 {
    Foo(bool),
    Bar,
}
enum NonEmptyEnum5 {
    V1, V2, V3, V4, V5,
}

fn empty_enum(x: EmptyEnum) {
    match x {} // ok
    match x {
        _ => {}, //~ ERROR unreachable pattern
    }
    match x {
        _ if false => {}, //~ ERROR unreachable pattern
    }
}

fn empty_foreign_enum(x: empty::EmptyForeignEnum) {
    match x {} // ok
    match x {
        _ => {}, //~ ERROR unreachable pattern
    }
    match x {
        _ if false => {}, //~ ERROR unreachable pattern
    }
}

fn never(x: !) {
    match x {} // ok
    match x {
        _ => {}, //~ ERROR unreachable pattern
    }
    match x {
        _ if false => {}, //~ ERROR unreachable pattern
    }
}

macro_rules! match_no_arms {
    ($e:expr) => {
        match $e {}
    };
}
macro_rules! match_guarded_arm {
    ($e:expr) => {
        match $e {
            _ if false => {}
        }
    };
}

fn main() {
    match_no_arms!(0u8); //~ ERROR type `u8` is non-empty
    match_no_arms!(NonEmptyStruct1); //~ ERROR type `NonEmptyStruct1` is non-empty
    match_no_arms!(NonEmptyStruct2(true)); //~ ERROR type `NonEmptyStruct2` is non-empty
    match_no_arms!((NonEmptyUnion1 { foo: () })); //~ ERROR type `NonEmptyUnion1` is non-empty
    match_no_arms!((NonEmptyUnion2 { foo: () })); //~ ERROR type `NonEmptyUnion2` is non-empty
    match_no_arms!(NonEmptyEnum1::Foo(true)); //~ ERROR `NonEmptyEnum1::Foo(_)` not covered
    match_no_arms!(NonEmptyEnum2::Foo(true)); //~ ERROR `NonEmptyEnum2::Foo(_)` and `NonEmptyEnum2::Bar` not covered
    match_no_arms!(NonEmptyEnum5::V1); //~ ERROR `NonEmptyEnum5::V1`, `NonEmptyEnum5::V2`, `NonEmptyEnum5::V3` and 2 more not covered

    match_guarded_arm!(0u8); //~ ERROR `_` not covered
    match_guarded_arm!(NonEmptyStruct1); //~ ERROR `NonEmptyStruct1` not covered
    match_guarded_arm!(NonEmptyStruct2(true)); //~ ERROR `NonEmptyStruct2(_)` not covered
    match_guarded_arm!((NonEmptyUnion1 { foo: () })); //~ ERROR `NonEmptyUnion1 { .. }` not covered
    match_guarded_arm!((NonEmptyUnion2 { foo: () })); //~ ERROR `NonEmptyUnion2 { .. }` not covered
    match_guarded_arm!(NonEmptyEnum1::Foo(true)); //~ ERROR `NonEmptyEnum1::Foo(_)` not covered
    match_guarded_arm!(NonEmptyEnum2::Foo(true)); //~ ERROR `NonEmptyEnum2::Foo(_)` and `NonEmptyEnum2::Bar` not covered
    match_guarded_arm!(NonEmptyEnum5::V1); //~ ERROR `NonEmptyEnum5::V1`, `NonEmptyEnum5::V2`, `NonEmptyEnum5::V3` and 2 more not covered
}
