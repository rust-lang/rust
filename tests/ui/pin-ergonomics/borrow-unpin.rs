//@ revisions: unpin pinned
#![feature(pin_ergonomics)]
#![allow(dead_code, incomplete_features)]

// For now, in order to ensure soundness, we move the place in `&pin mut place`
// if `place` is not `Unpin`.
// In the next step, we borrow the place instead of moving it, after that we
// have to makes sure `&pin mut place` and `&pin const place` cannot violate
// the mut-xor-share rules.

use std::marker::PhantomPinned;
use std::pin::Pin;

#[cfg(pinned)]
#[derive(Default)]
struct Foo(PhantomPinned);

#[cfg(unpin)]
#[derive(Default)]
struct Foo;

fn foo_mut(_: &mut Foo) {}
fn foo_ref(_: &Foo) {}
fn foo_pin_mut(_: &pin mut Foo) {}
fn foo_pin_ref(_: &pin const Foo) {}
fn foo_move(_: Foo) {}

fn immutable_pin_mut_then_move() {
    let mut foo = Foo::default();
    foo_pin_mut(&pin mut foo);
    foo_move(foo);

    let mut foo = Foo::default();
    let x = &pin mut foo;
    foo_move(foo);
    //~^ ERROR cannot move out of `foo` because it is borrowed
    foo_pin_mut(x); //
}

fn pin_mut_then_move() {
    let mut foo = Foo::default();
    foo_pin_mut(&pin mut foo); // ok
    foo_move(foo);

    let mut foo = Foo::default();
    let x = &pin mut foo; // ok
    foo_move(foo);
    //~^ ERROR cannot move out of `foo` because it is borrowed
    foo_pin_mut(x); //
}

fn pin_ref_then_move() {
    let foo = Foo::default();
    foo_pin_ref(&pin const foo); // ok
    foo_move(foo); // ok

    let foo = Foo::default();
    let x = &pin const foo; // ok
    foo_move(foo);
    //~^ ERROR cannot move out of `foo` because it is borrowed
    foo_pin_ref(x);
}

fn pin_mut_then_ref() {
    let mut foo = Foo::default();
    foo_pin_mut(&pin mut foo); // ok
    foo_ref(&foo);

    let mut foo = Foo::default();
    let x = &pin mut foo; // ok
    foo_ref(&foo);
    //~^ ERROR cannot borrow `foo` as immutable because it is also borrowed as mutable
    foo_pin_mut(x);
}

fn pin_ref_then_ref() {
    let mut foo = Foo::default();
    foo_pin_ref(&pin const foo); // ok
    foo_ref(&foo); // ok

    let mut foo = Foo::default();
    let x = &pin const foo; // ok
    foo_ref(&foo); // ok
    foo_pin_ref(x);
}

fn pin_mut_then_pin_mut() {
    let mut foo = Foo::default();
    foo_pin_mut(&pin mut foo); // ok
    foo_pin_mut(&pin mut foo);

    let mut foo = Foo::default();
    let x = &pin mut foo; // ok
    foo_pin_mut(&pin mut foo);
    //~^ ERROR cannot borrow `foo` as mutable more than once at a time
    foo_pin_mut(x);
}

fn pin_ref_then_pin_mut() {
    let mut foo = Foo::default();
    foo_pin_ref(&pin const foo); // ok
    foo_pin_mut(&pin mut foo); // ok

    let mut foo = Foo::default();
    let x = &pin const foo; // ok
    foo_pin_mut(&pin mut foo);
    //~^ ERROR cannot borrow `foo` as mutable because it is also borrowed as immutable
    foo_pin_ref(x);
}

fn pin_mut_then_pin_ref() {
    let mut foo = Foo::default();
    foo_pin_mut(&pin mut foo); // ok
    foo_pin_ref(&pin const foo);

    let mut foo = Foo::default();
    let x = &pin mut foo; // ok
    foo_pin_ref(&pin const foo);
    //~^ ERROR cannot borrow `foo` as immutable because it is also borrowed as mutable
    foo_pin_mut(x);
}

fn pin_ref_then_pin_ref() {
    let mut foo = Foo::default();
    foo_pin_ref(&pin const foo); // ok
    foo_pin_ref(&pin const foo); // ok

    let mut foo = Foo::default();
    let x = &pin const foo; // ok
    foo_pin_ref(&pin const foo); // ok
    foo_pin_ref(x);
}

fn main() {}
