// aux-build:empty.rs
// revisions: normal exhaustive_patterns
//
// This tests a match with no arms on various types.
#![feature(never_type)]
#![feature(never_type_fallback)]
#![cfg_attr(exhaustive_patterns, feature(exhaustive_patterns))]
#![deny(unreachable_patterns)]
//~^ NOTE the lint level is defined here

extern crate empty;

enum EmptyEnum {}

struct NonEmptyStruct1;
//~^ NOTE `NonEmptyStruct1` defined here
//~| NOTE `NonEmptyStruct1` defined here
struct NonEmptyStruct2(bool);
//~^ NOTE `NonEmptyStruct2` defined here
//~| NOTE `NonEmptyStruct2` defined here
union NonEmptyUnion1 {
    //~^ NOTE `NonEmptyUnion1` defined here
    //~| NOTE `NonEmptyUnion1` defined here
    foo: (),
}
union NonEmptyUnion2 {
    //~^ NOTE `NonEmptyUnion2` defined here
    //~| NOTE `NonEmptyUnion2` defined here
    foo: (),
    bar: (),
}
enum NonEmptyEnum1 {
    Foo(bool),
    //~^ NOTE `NonEmptyEnum1` defined here
    //~| NOTE `NonEmptyEnum1` defined here
    //~| NOTE not covered
    //~| NOTE not covered
}
enum NonEmptyEnum2 {
    Foo(bool),
    //~^ NOTE `NonEmptyEnum2` defined here
    //~| NOTE `NonEmptyEnum2` defined here
    //~| NOTE not covered
    //~| NOTE not covered
    Bar,
    //~^ NOTE not covered
    //~| NOTE not covered
}
enum NonEmptyEnum5 {
    //~^ NOTE `NonEmptyEnum5` defined here
    //~| NOTE `NonEmptyEnum5` defined here
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

fn empty_foreign_enum_private(x: Option<empty::SecretlyUninhabitedForeignStruct>) {
    let None = x;
    //~^ ERROR refutable pattern in local binding
    //~| NOTE `let` bindings require an "irrefutable pattern"
    //~| NOTE for more information, visit
    //~| NOTE the matched value is of type
    //~| NOTE pattern `Some(_)` not covered
    //[exhaustive_patterns]~| NOTE currently uninhabited, but this variant contains private fields
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
                         //~| NOTE the matched value is of type
    match_no_arms!(NonEmptyStruct1); //~ ERROR type `NonEmptyStruct1` is non-empty
                                     //~| NOTE the matched value is of type
    match_no_arms!(NonEmptyStruct2(true)); //~ ERROR type `NonEmptyStruct2` is non-empty
                                           //~| NOTE the matched value is of type
    match_no_arms!((NonEmptyUnion1 { foo: () })); //~ ERROR type `NonEmptyUnion1` is non-empty
                                                  //~| NOTE the matched value is of type
    match_no_arms!((NonEmptyUnion2 { foo: () })); //~ ERROR type `NonEmptyUnion2` is non-empty
                                                  //~| NOTE the matched value is of type
    match_no_arms!(NonEmptyEnum1::Foo(true)); //~ ERROR `NonEmptyEnum1::Foo(_)` not covered
                                              //~| NOTE pattern `NonEmptyEnum1::Foo(_)` not covered
                                              //~| NOTE the matched value is of type
    match_no_arms!(NonEmptyEnum2::Foo(true)); //~ ERROR `NonEmptyEnum2::Foo(_)` and `NonEmptyEnum2::Bar` not covered
                                              //~| NOTE patterns `NonEmptyEnum2::Foo(_)` and
                                              //~| NOTE the matched value is of type
    match_no_arms!(NonEmptyEnum5::V1); //~ ERROR `NonEmptyEnum5::V1`, `NonEmptyEnum5::V2`, `NonEmptyEnum5::V3` and 2 more not covered
                                       //~| NOTE patterns `NonEmptyEnum5::V1`, `NonEmptyEnum5::V2`
                                       //~| NOTE the matched value is of type

    match_guarded_arm!(0u8); //~ ERROR `_` not covered
                             //~| NOTE the matched value is of type
                             //~| NOTE match arms with guards don't count towards exhaustivity
                             //~| NOTE pattern `_` not covered
                             //~| NOTE in this expansion of match_guarded_arm!
    match_guarded_arm!(NonEmptyStruct1); //~ ERROR `NonEmptyStruct1` not covered
                                         //~| NOTE pattern `NonEmptyStruct1` not covered
                                         //~| NOTE the matched value is of type
                                         //~| NOTE match arms with guards don't count towards exhaustivity
                                         //~| NOTE in this expansion of match_guarded_arm!
    match_guarded_arm!(NonEmptyStruct2(true)); //~ ERROR `NonEmptyStruct2(_)` not covered
                                               //~| NOTE the matched value is of type
                                               //~| NOTE pattern `NonEmptyStruct2(_)` not covered
                                               //~| NOTE match arms with guards don't count towards exhaustivity
                                               //~| NOTE in this expansion of match_guarded_arm!
    match_guarded_arm!((NonEmptyUnion1 { foo: () })); //~ ERROR `NonEmptyUnion1 { .. }` not covered
                                                      //~| NOTE the matched value is of type
                                                      //~| NOTE pattern `NonEmptyUnion1 { .. }` not covered
                                                      //~| NOTE match arms with guards don't count towards exhaustivity
                                                      //~| NOTE in this expansion of match_guarded_arm!
    match_guarded_arm!((NonEmptyUnion2 { foo: () })); //~ ERROR `NonEmptyUnion2 { .. }` not covered
                                                      //~| NOTE the matched value is of type
                                                      //~| NOTE pattern `NonEmptyUnion2 { .. }` not covered
                                                      //~| NOTE match arms with guards don't count towards exhaustivity
                                                      //~| NOTE in this expansion of match_guarded_arm!
    match_guarded_arm!(NonEmptyEnum1::Foo(true)); //~ ERROR `NonEmptyEnum1::Foo(_)` not covered
                                                  //~| NOTE the matched value is of type
                                                  //~| NOTE pattern `NonEmptyEnum1::Foo(_)` not covered
                                                  //~| NOTE match arms with guards don't count towards exhaustivity
                                                  //~| NOTE in this expansion of match_guarded_arm!
    match_guarded_arm!(NonEmptyEnum2::Foo(true)); //~ ERROR `NonEmptyEnum2::Foo(_)` and `NonEmptyEnum2::Bar` not covered
                                                  //~| NOTE the matched value is of type
                                                  //~| NOTE patterns `NonEmptyEnum2::Foo(_)` and
                                                  //~| NOTE match arms with guards don't count towards exhaustivity
                                                  //~| NOTE in this expansion of match_guarded_arm!
    match_guarded_arm!(NonEmptyEnum5::V1); //~ ERROR `NonEmptyEnum5::V1`, `NonEmptyEnum5::V2`, `NonEmptyEnum5::V3` and 2 more not covered
                                           //~| NOTE the matched value is of type
                                           //~| NOTE patterns `NonEmptyEnum5::V1`,
                                           //~| NOTE match arms with guards don't count towards exhaustivity
                                           //~| NOTE in this expansion of match_guarded_arm!
}
