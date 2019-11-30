#![feature(never_type)]
#![feature(exhaustive_patterns)]
#![deny(unreachable_patterns)]
enum Foo {}

struct NonEmptyStruct(bool);
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

fn foo(x: Foo) {
    match x {} // ok
    match x {
        _ => {}, //~ ERROR unreachable pattern
    }
}

fn main() {
    // `exhaustive_patterns` is not on, so uninhabited branches are not detected as unreachable.
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
    match NonEmptyEnum1::Foo(true) {}
    //~^ ERROR type `NonEmptyEnum1` is non-empty
    match NonEmptyEnum2::Foo(true) {}
    //~^ ERROR type `NonEmptyEnum2` is non-empty
    match NonEmptyEnum5::V1 {}
    //~^ ERROR type `NonEmptyEnum5` is non-empty
}
