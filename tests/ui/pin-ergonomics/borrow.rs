#![feature(pin_ergonomics)]
#![allow(dead_code, incomplete_features)]

// Makes sure we can handle `&pin mut place` and `&pin const place` as sugar for
// `std::pin::pin!(place)` and `Pin::new(&place)`.

use std::pin::Pin;

struct Foo;

fn foo_pin_mut(_: Pin<&mut Foo>) {
}

fn foo_pin_ref(_: Pin<&Foo>) {
}

fn bar() {
    let mut x: Pin<&mut _> = &pin mut Foo;
    foo_pin_mut(x.as_mut());
    foo_pin_mut(x.as_mut());
    foo_pin_ref(x);

    let x: Pin<&_> = &pin const Foo;

    foo_pin_ref(x);
    foo_pin_ref(x);
}

fn baz(mut x: Foo, y: Foo) {
    // For now, `x` is moved to ensure soundness of creating this `Pin<&mut Foo>`.
    let _x = &pin mut x;
    let _x = x; //~ ERROR use of moved value: `x`

    let _y = &pin const y;
    let _y = y; // ok because `&pin const y` dosn't move `y`
}

fn main() {}
