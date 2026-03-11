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
    let _x = &pin mut x;
    let _x = x; //~ ERROR cannot move out of `x` because it is borrowed [E0505]
    let _x = &mut x; //~ ERROR cannot borrow `x` as mutable more than once at a time [E0499]
    //~^ ERROR borrow of moved value: `x` [E0382]

    x = Foo;
    let _x = &mut x; // ok

    let _y = &pin const y;
    let _y = y; //~ ERROR cannot move out of `y` because it is borrowed [E0505]
    let _y = &mut y; //~ ERROR cannot borrow `y` as mutable because it is also borrowed as immutable [E0502]
    //~^ ERROR borrow of moved value: `y` [E0382]
    //~| ERROR cannot borrow `y` as mutable, as it is not declared as mutable [E0596]
}

fn main() {}
