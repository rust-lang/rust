//@ edition:2024
#![feature(pin_ergonomics)]
#![allow(incomplete_features)]

// Regression test for #157634.
//
// The implicit pin-projection (via match ergonomics) is only allowed on `#[pin_v2]` types, but
// the explicit `&pin mut` / `ref pin` pattern forms used to project through *any* type. That is
// unsound: it lets safe code form a `Pin<&mut Field>` for a type that never opted into structural
// pinning, breaking the `Pin` guarantee. Check that the explicit forms are now gated the same way
// as the implicit one.

use std::pin::Pin;

struct NotPinProject<T>(T);

struct NotPinProjectStruct<T> {
    x: T,
}

enum NotPinProjectEnum<T> {
    Tuple(T),
    Struct { x: T },
}

fn tuple_struct<T>(p: Pin<&mut NotPinProject<T>>) {
    let &pin mut NotPinProject(ref pin mut _x) = p;
    //~^ ERROR cannot project on type that is not `#[pin_v2]`
}

fn struct_field<T>(p: Pin<&mut NotPinProjectStruct<T>>) {
    let &pin mut NotPinProjectStruct { x: ref pin mut _x } = p;
    //~^ ERROR cannot project on type that is not `#[pin_v2]`
}

fn shared<T>(p: Pin<&NotPinProject<T>>) {
    let &pin const NotPinProject(ref pin const _x) = p;
    //~^ ERROR cannot project on type that is not `#[pin_v2]`
}

fn enum_tuple<T>(p: Pin<&mut NotPinProjectEnum<T>>) {
    if let &pin mut NotPinProjectEnum::Tuple(ref pin mut _x) = p {}
    //~^ ERROR cannot project on type that is not `#[pin_v2]`
}

fn enum_struct<T>(p: Pin<&mut NotPinProjectEnum<T>>) {
    if let &pin mut NotPinProjectEnum::Struct { x: ref pin mut _x } = p {}
    //~^ ERROR cannot project on type that is not `#[pin_v2]`
}

// The exact shape from the issue: `Thing` unconditionally implements `Unpin`, so it must not be
// possible to project a pinned reference to one of its fields.
struct Thing<T>(T);
impl<T> Unpin for Thing<T> {}

fn issue_157634<T>(pinned_thing: Pin<&mut Thing<Option<T>>>) {
    let &pin mut Thing(ref pin mut _pinned_option) = pinned_thing;
    //~^ ERROR cannot project on type that is not `#[pin_v2]`
}

fn main() {}
