#![feature(never_type)]
#![deny(unreachable_patterns)]
enum Foo {}

struct NonEmptyStruct(bool); //~ `NonEmptyStruct` defined here
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

fn foo1(x: Foo) {
    match x {} // ok
}

fn foo2(x: Foo) {
    match x {
        _ => {}, // FIXME: should be unreachable
    }
}

fn main() {
    // `exhaustive_patterns` is not on, so uninhabited branches are not detected as unreachable.
    match None::<!> {
        None => {}
        Some(_) => {}
    }
    match None::<Foo> {
        None => {}
        Some(_) => {}
    }

    match 0u8 {}
    //~^ ERROR type `u8` is non-empty
    match NonEmptyStruct(true) {}
    //~^ ERROR pattern `NonEmptyStruct` of type `NonEmptyStruct` is not handled
    match NonEmptyEnum1::Foo(true) {}
    //~^ ERROR pattern `Foo` of type `NonEmptyEnum1` is not handled
    match NonEmptyEnum2::Foo(true) {}
    //~^ ERROR multiple patterns of type `NonEmptyEnum2` are not handled
    match NonEmptyEnum5::V1 {}
    //~^ ERROR multiple patterns of type `NonEmptyEnum5` are not handled
}
