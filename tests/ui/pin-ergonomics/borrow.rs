#![feature(pin_ergonomics)]

// Makes sure we can handle `&pin mut place` and `&pin const place` as sugar for
// `std::pin::pin!(place)` and `Pin::new(&place)`.

use std::marker::PhantomPinned;
use std::pin::Pin;

#[pin_v2]
#[derive(Default)]
struct Foo(PhantomPinned);

fn foo_pin_mut(_: Pin<&mut Foo>) {}

fn foo_pin_ref(_: Pin<&Foo>) {}

fn bar() {
    let mut x: Pin<&mut _> = &pin mut Foo::default();
    foo_pin_mut(x.as_mut());
    foo_pin_mut(x.as_mut());
    foo_pin_ref(x);

    let x: Pin<&_> = &pin const Foo::default();

    foo_pin_ref(x);
    foo_pin_ref(x);
}

fn baz(mut x: Foo, mut y: Foo) {
    {
        let _x = &pin mut x;
    }
    let _x = &mut x; //~ ERROR cannot borrow `x` as mutable because it is pinned
    let _x = x; //~ ERROR cannot move out of `x` because it is pinned

    x = Foo::default();
    let _x = &mut x; // ok

    {
        let _y = &pin const y;
    }
    let _y = &mut y; //~ ERROR cannot borrow `y` as mutable because it is pinned
    let _y = y; //~ ERROR cannot move out of `y` because it is pinned
}

fn ref_pin_mut_then_mut_borrow_and_move(mut x: Foo) {
    {
        let ref pin mut _x = x;
    }
    let _x = &mut x; //~ ERROR cannot borrow `x` as mutable because it is pinned
    let _x = x; //~ ERROR cannot move out of `x` because it is pinned
}

fn ref_pin_const_then_mut_borrow_and_move(mut y: Foo) {
    {
        let ref pin const _y = y;
    }
    let _y = &mut y; //~ ERROR cannot borrow `y` as mutable because it is pinned
    let _y = y; //~ ERROR cannot move out of `y` because it is pinned
}

fn main() {}
