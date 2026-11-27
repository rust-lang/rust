//@ revisions: normal pin_ergonomics deref_patterns both
//@ edition:2024
#![cfg_attr(any(pin_ergonomics, both), feature(pin_ergonomics))]
#![cfg_attr(any(deref_patterns, both), feature(deref_patterns))]
#![allow(incomplete_features)]

// This test verifies that the `pin_ergonomics` feature works well
// together with the `deref_patterns` feature under the error:
// "mix of deref patterns and normal constructors".

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

    match foo.as_mut() {
        Foo { x, y } => {}
        //[normal]~^ ERROR mismatched types
        //[deref_patterns]~^^ ERROR mix of deref patterns and normal constructors
        //[pin_ergonomics]~^^^ ERROR mix of deref patterns and normal constructors
        //[both]~^^^^ ERROR mix of deref patterns and normal constructors
        Pin { .. } => {}
    }
    let _ = || match foo.as_mut() {
        Foo { x, y } => {}
        //[normal]~^ ERROR mismatched types
        //[deref_patterns]~^^ ERROR mix of deref patterns and normal constructors
        //[pin_ergonomics]~^^^ ERROR mix of deref patterns and normal constructors
        //[both]~^^^^ ERROR mix of deref patterns and normal constructors
        Pin { .. } => {}
    };
}

fn foo_const<T: Unpin, U: Unpin>(foo: Pin<&Foo<T, U>>) {
    let Foo { x, y } = foo;
    //[normal]~^ ERROR mismatched types

    match foo {
        Foo { x, y } => {}
        //[normal]~^ ERROR mismatched types
        //[deref_patterns]~^^ ERROR mix of deref patterns and normal constructors
        //[pin_ergonomics]~^^^ ERROR mix of deref patterns and normal constructors
        //[both]~^^^^ ERROR mix of deref patterns and normal constructors
        Pin { .. } => {}
    }
    let _ = || match foo {
        Foo { x, y } => {}
        //[normal]~^ ERROR mismatched types
        //[deref_patterns]~^^ ERROR mix of deref patterns and normal constructors
        //[pin_ergonomics]~^^^ ERROR mix of deref patterns and normal constructors
        //[both]~^^^^ ERROR mix of deref patterns and normal constructors
        Pin { .. } => {}
    };
}

fn bar_mut<T: Unpin, U: Unpin>(bar: Pin<&mut Bar<T, U>>) {
    let Bar(x, y) = bar.as_mut();
    //[normal]~^ ERROR mismatched types

    match bar.as_mut() {
        Bar(x, y) => {}
        //[normal]~^ ERROR mismatched types
        //[deref_patterns]~^^ ERROR mix of deref patterns and normal constructors
        //[pin_ergonomics]~^^^ ERROR mix of deref patterns and normal constructors
        //[both]~^^^^ ERROR mix of deref patterns and normal constructors
        Pin { .. } => {}
    }
    let _ = || match bar.as_mut() {
        Bar(x, y) => {}
        //[normal]~^ ERROR mismatched types
        //[deref_patterns]~^^ ERROR mix of deref patterns and normal constructors
        //[pin_ergonomics]~^^^ ERROR mix of deref patterns and normal constructors
        //[both]~^^^^ ERROR mix of deref patterns and normal constructors
        Pin { .. } => {}
    };
}

fn bar_const<T: Unpin, U: Unpin>(bar: Pin<&Bar<T, U>>) {
    let Bar(x, y) = bar;
    //[normal]~^ ERROR mismatched types

    match bar {
        Bar(x, y) => {}
        //[normal]~^ ERROR mismatched types
        //[deref_patterns]~^^ ERROR mix of deref patterns and normal constructors
        //[pin_ergonomics]~^^^ ERROR mix of deref patterns and normal constructors
        //[both]~^^^^ ERROR mix of deref patterns and normal constructors
        Pin { .. } => {}
    }
    let _ = || match bar {
        Bar(x, y) => {}
        //[normal]~^ ERROR mismatched types
        //[deref_patterns]~^^ ERROR mix of deref patterns and normal constructors
        //[pin_ergonomics]~^^^ ERROR mix of deref patterns and normal constructors
        //[both]~^^^^ ERROR mix of deref patterns and normal constructors
        Pin { .. } => {}
    };
}

fn non_pin_project<T, U: Unpin>(foo: Pin<&mut NonPinProject<T>>, bar: Pin<&NonPinProject<U>>) {
    let NonPinProject { x } = foo;
    //[normal]~^ ERROR mismatched types
    //[pin_ergonomics]~^^ ERROR cannot project on type that is not `#[pin_v2]`
    //[both]~^^^ ERROR cannot project on type that is not `#[pin_v2]`
    let NonPinProject { x } = bar;
    //[normal]~^ ERROR mismatched types
    //[pin_ergonomics]~^^ ERROR cannot project on type that is not `#[pin_v2]`
    //[both]~^^^ ERROR cannot project on type that is not `#[pin_v2]`

    match foo {
        NonPinProject { x } => {}
        //[normal]~^ ERROR mismatched types
        //[deref_patterns]~^^ ERROR mix of deref patterns and normal constructors
        //[pin_ergonomics]~^^^ ERROR cannot project on type that is not `#[pin_v2]`
        //[both]~^^^^ ERROR cannot project on type that is not `#[pin_v2]`
        Pin { .. } => {}
    }
    match bar {
        NonPinProject { x } => {}
        //[normal]~^ ERROR mismatched types
        //[deref_patterns]~^^ ERROR mix of deref patterns and normal constructors
        //[pin_ergonomics]~^^^ ERROR cannot project on type that is not `#[pin_v2]`
        //[both]~^^^^ ERROR cannot project on type that is not `#[pin_v2]`
        Pin { .. } => {}
    }
}

fn main() {}
