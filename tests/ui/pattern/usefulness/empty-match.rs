//@ revisions: normal exhaustive_patterns
//
// This tests a match with no arms on various types.
#![feature(never_type)]
#![cfg_attr(exhaustive_patterns, feature(exhaustive_patterns))]
#![deny(unreachable_patterns)]

fn nonempty<const N: usize>(arrayN_of_empty: [!; N]) {
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

    struct NonEmptyStruct1;
    struct NonEmptyStruct2(bool);
    union NonEmptyUnion1 {
        foo: (),
    }
    union NonEmptyUnion2 {
        foo: (),
        bar: !,
    }
    enum NonEmptyEnum1 {
        Foo(bool),
    }
    enum NonEmptyEnum2 {
        Foo(bool),
        Bar,
    }
    enum NonEmptyEnum5 {
        V1,
        V2,
        V3,
        V4,
        V5,
    }
    let array0_of_empty: [!; 0] = [];

    match_no_arms!(0u8); //~ ERROR type `u8` is non-empty
    match_no_arms!(0i8); //~ ERROR type `i8` is non-empty
    match_no_arms!(0usize); //~ ERROR type `usize` is non-empty
    match_no_arms!(0isize); //~ ERROR type `isize` is non-empty
    match_no_arms!(NonEmptyStruct1); //~ ERROR type `NonEmptyStruct1` is non-empty
    match_no_arms!(NonEmptyStruct2(true)); //~ ERROR type `NonEmptyStruct2` is non-empty
    match_no_arms!((NonEmptyUnion1 { foo: () })); //~ ERROR type `NonEmptyUnion1` is non-empty
    match_no_arms!((NonEmptyUnion2 { foo: () })); //~ ERROR type `NonEmptyUnion2` is non-empty
    match_no_arms!(NonEmptyEnum1::Foo(true)); //~ ERROR `NonEmptyEnum1::Foo(_)` not covered
    match_no_arms!(NonEmptyEnum2::Foo(true)); //~ ERROR `NonEmptyEnum2::Foo(_)` and `NonEmptyEnum2::Bar` not covered
    match_no_arms!(NonEmptyEnum5::V1); //~ ERROR `NonEmptyEnum5::V1`, `NonEmptyEnum5::V2`, `NonEmptyEnum5::V3` and 2 more not covered
    match_no_arms!(array0_of_empty); //~ ERROR type `[!; 0]` is non-empty
    match_no_arms!(arrayN_of_empty); //~ ERROR type `[!; N]` is non-empty

    match_guarded_arm!(0u8); //~ ERROR `0_u8..=u8::MAX` not covered
    match_guarded_arm!(0i8); //~ ERROR `i8::MIN..=i8::MAX` not covered
    match_guarded_arm!(0usize); //~ ERROR `0_usize..` not covered
    match_guarded_arm!(0isize); //~ ERROR `_` not covered
    match_guarded_arm!(NonEmptyStruct1); //~ ERROR `NonEmptyStruct1` not covered
    match_guarded_arm!(NonEmptyStruct2(true)); //~ ERROR `NonEmptyStruct2(_)` not covered
    match_guarded_arm!((NonEmptyUnion1 { foo: () })); //~ ERROR `NonEmptyUnion1 { .. }` not covered
    match_guarded_arm!((NonEmptyUnion2 { foo: () })); //~ ERROR `NonEmptyUnion2 { .. }` not covered
    match_guarded_arm!(NonEmptyEnum1::Foo(true)); //~ ERROR `NonEmptyEnum1::Foo(_)` not covered
    match_guarded_arm!(NonEmptyEnum2::Foo(true)); //~ ERROR `NonEmptyEnum2::Foo(_)` and `NonEmptyEnum2::Bar` not covered
    match_guarded_arm!(NonEmptyEnum5::V1); //~ ERROR `NonEmptyEnum5::V1`, `NonEmptyEnum5::V2`, `NonEmptyEnum5::V3` and 2 more not covered
    match_guarded_arm!(array0_of_empty); //~ ERROR `[]` not covered
    match_guarded_arm!(arrayN_of_empty); //~ ERROR `[]` not covered
}

fn main() {}
