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

fn foo(x: Foo) {
    match x {} // ok
    match x {
        _ => {}, //~ ERROR unreachable pattern
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

    match 0u8 {}
    //~^ ERROR type `u8` is non-empty
    match NonEmptyStruct(true) {}
    //~^ ERROR type `NonEmptyStruct` is non-empty
    match (NonEmptyUnion1 { foo: () }) {}
    //~^ ERROR type `NonEmptyUnion1` is non-empty
    match (NonEmptyUnion2 { foo: () }) {}
    //~^ ERROR type `NonEmptyUnion2` is non-empty
    match NonEmptyEnum1::Foo(true) {}
    //~^ ERROR pattern `Foo` of type `NonEmptyEnum1` is not handled
    match NonEmptyEnum2::Foo(true) {}
    //~^ ERROR multiple patterns of type `NonEmptyEnum2` are not handled
    match NonEmptyEnum5::V1 {}
    //~^ ERROR multiple patterns of type `NonEmptyEnum5` are not handled
}
