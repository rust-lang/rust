//@ revisions: normal pin_ergonomics deref_patterns both
//@ edition:2024
#![cfg_attr(any(pin_ergonomics, both), feature(pin_ergonomics))]
#![cfg_attr(any(deref_patterns, both), feature(deref_patterns))]
#![allow(incomplete_features)]

//! This test used to verify that the `pin_ergonomics` feature works well
//! together with the `deref_patterns` feature.

use std::pin::Pin;

#[cfg_attr(any(pin_ergonomics, both), pin_v2)]
struct Foo<T, U> {
    x: T,
    y: U,
}

#[cfg_attr(any(pin_ergonomics, both), pin_v2)]
struct Bar<T, U>(T, U);

#[cfg_attr(any(pin_ergonomics, both), pin_v2)]
enum Baz<T, U> {
    Foo(T, U),
    Bar { x: T, y: U },
}

struct NonPinProject<T> {
    x: T,
}

fn foo_mut<T: Unpin, U: Unpin>(mut foo: Pin<&mut Foo<T, U>>) {
    let Foo { x, y } = foo.as_mut();
    //[normal]~^ ERROR mismatched types
    //[deref_patterns]~^^ ERROR E0507

    match foo.as_mut() {
        //[deref_patterns]~^ ERROR E0507
        Foo { x, y } => {}
        //[normal]~^ ERROR mismatched types
        Pin { .. } => {}
    }
    let _ = || match foo.as_mut() {
        //[deref_patterns]~^ ERROR E0507
        Foo { x, y } => {}
        //[normal]~^ ERROR mismatched types
        Pin { .. } => {}
    };
}

fn foo_const<T: Unpin, U: Unpin>(foo: Pin<&Foo<T, U>>) {
    let Foo { x, y } = foo;
    //[normal]~^ ERROR mismatched types
    //[deref_patterns]~^^ ERROR E0507

    match foo {
        //[deref_patterns]~^ ERROR E0507
        Foo { x, y } => {}
        //[normal]~^ ERROR mismatched types
        Pin { .. } => {}
    }
    let _ = || match foo {
        //[deref_patterns]~^ ERROR E0507
        Foo { x, y } => {}
        //[normal]~^ ERROR mismatched types
        Pin { .. } => {}
    };
}

fn bar_mut<T: Unpin, U: Unpin>(mut bar: Pin<&mut Bar<T, U>>) {
    let Bar(x, y) = bar.as_mut();
    //[normal]~^ ERROR mismatched types
    //[deref_patterns]~^^ ERROR E0507

    match bar.as_mut() {
        //[deref_patterns]~^ ERROR E0507
        Bar(x, y) => {}
        //[normal]~^ ERROR mismatched types
        Pin { .. } => {}
    }
    let _ = || match bar.as_mut() {
        //[deref_patterns]~^ ERROR E0507
        Bar(x, y) => {}
        //[normal]~^ ERROR mismatched types
        Pin { .. } => {}
    };
}

fn bar_const<T: Unpin, U: Unpin>(bar: Pin<&Bar<T, U>>) {
    let Bar(x, y) = bar;
    //[normal]~^ ERROR mismatched types
    //[deref_patterns]~^^ ERROR E0507

    match bar {
        //[deref_patterns]~^ ERROR E0507
        Bar(x, y) => {}
        //[normal]~^ ERROR mismatched types
        Pin { .. } => {}
    }
    let _ = || match bar {
        //[deref_patterns]~^ ERROR E0507
        Bar(x, y) => {}
        //[normal]~^ ERROR mismatched types
        Pin { .. } => {}
    };
}

fn non_pin_project<T, U: Unpin>(foo: Pin<&mut NonPinProject<T>>, bar: Pin<&NonPinProject<U>>) {
    let NonPinProject { x } = foo;
    //[normal]~^ ERROR mismatched types
    //[deref_patterns]~^^ ERROR E0507
    //[pin_ergonomics]~^^^ ERROR cannot project on type that is not `#[pin_v2]`
    //[both]~^^^^ ERROR cannot project on type that is not `#[pin_v2]`
    let NonPinProject { x } = bar;
    //[normal]~^ ERROR mismatched types
    //[deref_patterns]~^^ ERROR E0507
    //[pin_ergonomics]~^^^ ERROR cannot project on type that is not `#[pin_v2]`
    //[both]~^^^^ ERROR cannot project on type that is not `#[pin_v2]`

    match foo {
        //[deref_patterns]~^ ERROR E0507
        NonPinProject { x } => {}
        //[normal]~^ ERROR mismatched types
        //[pin_ergonomics]~^^ ERROR cannot project on type that is not `#[pin_v2]`
        //[both]~^^^ ERROR cannot project on type that is not `#[pin_v2]`
        Pin { .. } => {}
    }
    match bar {
        //[deref_patterns]~^ ERROR E0507
        NonPinProject { x } => {}
        //[normal]~^ ERROR mismatched types
        //[pin_ergonomics]~^^ ERROR cannot project on type that is not `#[pin_v2]`
        //[both]~^^^ ERROR cannot project on type that is not `#[pin_v2]`
        Pin { .. } => {}
    }
}

fn main() {}
