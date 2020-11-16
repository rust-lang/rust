// aux-build:empty.rs
#![feature(never_type)]
#![feature(never_type_fallback)]
#![deny(unreachable_patterns)]

extern crate empty;

enum EmptyEnum {}

struct NonEmptyStruct(bool); //~ `NonEmptyStruct` defined here
union NonEmptyUnion1 { //~ `NonEmptyUnion1` defined here
    foo: (),
}
union NonEmptyUnion2 { //~ `NonEmptyUnion2` defined here
    foo: (),
    bar: (),
}
enum NonEmptyEnum1 { //~ `NonEmptyEnum1` defined here
    Foo(bool),
    //~^ not covered
    //~| not covered
}
enum NonEmptyEnum2 { //~ `NonEmptyEnum2` defined here
    Foo(bool),
    //~^ not covered
    //~| not covered
    Bar,
    //~^ not covered
    //~| not covered
}
enum NonEmptyEnum5 { //~ `NonEmptyEnum5` defined here
    V1, V2, V3, V4, V5,
}

macro_rules! match_empty {
    ($e:expr) => {
        match $e {}
    };
}
macro_rules! match_false {
    ($e:expr) => {
        match $e {
            _ if false => {}
        }
    };
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

fn main() {
    // `exhaustive_patterns` is not on, so uninhabited branches are not detected as unreachable.
    match None::<!> {
        None => {}
        Some(_) => {}
    }
    match None::<EmptyEnum> {
        None => {}
        Some(_) => {}
    }

    match_empty!(0u8);
    //~^ ERROR type `u8` is non-empty
    match_empty!(NonEmptyStruct(true));
    //~^ ERROR type `NonEmptyStruct` is non-empty
    match_empty!((NonEmptyUnion1 { foo: () }));
    //~^ ERROR type `NonEmptyUnion1` is non-empty
    match_empty!((NonEmptyUnion2 { foo: () }));
    //~^ ERROR type `NonEmptyUnion2` is non-empty
    match_empty!(NonEmptyEnum1::Foo(true));
    //~^ ERROR `Foo(_)` not covered
    match_empty!(NonEmptyEnum2::Foo(true));
    //~^ ERROR `Foo(_)` and `Bar` not covered
    match_empty!(NonEmptyEnum5::V1);
    //~^ ERROR `V1`, `V2`, `V3` and 2 more not covered

    match_false!(0u8);
    //~^ ERROR `_` not covered
    match_false!(NonEmptyStruct(true));
    //~^ ERROR `NonEmptyStruct(_)` not covered
    match_false!((NonEmptyUnion1 { foo: () }));
    //~^ ERROR `NonEmptyUnion1 { .. }` not covered
    match_false!((NonEmptyUnion2 { foo: () }));
    //~^ ERROR `NonEmptyUnion2 { .. }` not covered
    match_false!(NonEmptyEnum1::Foo(true));
    //~^ ERROR `Foo(_)` not covered
    match_false!(NonEmptyEnum2::Foo(true));
    //~^ ERROR `Foo(_)` and `Bar` not covered
    match_false!(NonEmptyEnum5::V1);
    //~^ ERROR `V1`, `V2`, `V3` and 2 more not covered
}
