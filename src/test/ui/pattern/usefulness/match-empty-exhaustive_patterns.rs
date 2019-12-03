#![feature(never_type)]
#![feature(exhaustive_patterns)]
#![deny(unreachable_patterns)]
enum Foo {}

struct NonEmptyStruct(bool);
union NonEmptyUnion1 {
    foo: (),
}
union NonEmptyUnion2 {
    foo: (),
    bar: (),
}
enum NonEmptyEnum1 { //~ `NonEmptyEnum1` defined here
    Foo(bool), //~ variant not covered
}
enum NonEmptyEnum2 { //~ `NonEmptyEnum2` defined here
    Foo(bool), //~ variant not covered
    Bar, //~ variant not covered
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

fn foo(x: Foo) {
    match_empty!(x); // ok
    match x {
        _ => {}, //~ ERROR unreachable pattern
    }
    match x {
        _ if false => {}, //~ ERROR unreachable pattern
    }
}

fn main() {
    match None::<!> {
        None => {}
        Some(_) => {} //~ ERROR unreachable pattern
    }
    match None::<Foo> {
        None => {}
        Some(_) => {} //~ ERROR unreachable pattern
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
    //~^ ERROR pattern `Foo` of type `NonEmptyEnum1` is not handled
    match_empty!(NonEmptyEnum2::Foo(true));
    //~^ ERROR multiple patterns of type `NonEmptyEnum2` are not handled
    match_empty!(NonEmptyEnum5::V1);
    //~^ ERROR multiple patterns of type `NonEmptyEnum5` are not handled

    match_false!(0u8);
    //~^ ERROR `_` not covered
    match_false!(NonEmptyStruct(true));
    //~^ ERROR `_` not covered
    match_false!((NonEmptyUnion1 { foo: () }));
    //~^ ERROR `_` not covered
    match_false!((NonEmptyUnion2 { foo: () }));
    //~^ ERROR `_` not covered
    match_false!(NonEmptyEnum1::Foo(true));
    //~^ ERROR `_` not covered
    match_false!(NonEmptyEnum2::Foo(true));
    //~^ ERROR `_` not covered
    match_false!(NonEmptyEnum5::V1);
    //~^ ERROR `_` not covered
}
