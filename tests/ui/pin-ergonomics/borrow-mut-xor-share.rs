#![feature(pin_ergonomics)]
#![allow(dead_code, incomplete_features)]

// For now, we move the place in `&pin mut place` to ensure soundness.
// In the next step, we borrow the place instead of moving it, after that we
// have to makes sure `&pin mut place` and `&pin const place` cannot violate
// the mut-xor-share rules.

use std::pin::Pin;

struct Foo;

fn foo_mut(_: &mut Foo) {
}

fn foo_ref(_: &Foo) {
}

fn foo_pin_mut(_: Pin<&mut Foo>) {
}

fn foo_pin_ref(_: Pin<&Foo>) {
}

fn bar() {
    let foo = Foo;
    foo_pin_mut(&pin mut foo); // ok because `foo` is moved`

    let mut foo = Foo;
    let x = &pin mut foo;
    foo_pin_ref(&pin const foo); //~ ERROR borrow of moved value: `foo`
    foo_pin_mut(&pin mut foo); // ok
    foo_ref(&foo); //~ ERROR borrow of moved value: `foo`
    foo_mut(&mut foo); // ok

    foo_pin_mut(x);

    let mut foo = Foo;
    let x = &pin const foo;
    foo_pin_ref(&pin const foo); // ok
    foo_pin_mut(&pin mut foo); //~ ERROR cannot move out of `foo` because it is borrowed
    foo_ref(&foo); //~ ERROR borrow of moved value: `foo`
    foo_mut(&mut foo); //~ ERROR cannot borrow `foo` as mutable because it is also borrowed as immutable

    foo_pin_ref(x);

    let mut foo = Foo;
    let x = &mut foo;
    foo_pin_ref(&pin const foo); //~ ERROR cannot borrow `foo` as immutable because it is also borrowed as mutable
    foo_pin_mut(&pin mut foo); //~ ERROR cannot move out of `foo` because it is borrowed

    foo_mut(x);

    let mut foo = Foo;
    let x = &foo;
    foo_pin_ref(&pin const foo); // ok
    foo_pin_mut(&pin mut foo); //~ ERROR cannot move out of `foo` because it is borrowed

    foo_ref(x);
}

fn main() {}
