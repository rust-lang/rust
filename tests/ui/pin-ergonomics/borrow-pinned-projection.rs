#![feature(pin_ergonomics)]
#![allow(incomplete_features)]

// This protects pinning projected places. Pinning `pair.0` must not pin
// unrelated disjoint fields, but it must reject later mutable borrows or moves
// of `pair.0` until reassignment.

struct Foo;

fn mutable_borrow_of_pinned_projection() {
    let mut pair = (Foo, Foo);

    {
        let _pin = &pin mut pair.0;
    }

    let _other = &mut pair.1;
    let _borrow = &mut pair.0;
    //~^ ERROR cannot borrow `pair.0` as mutable because it is pinned
}

fn move_of_pinned_projection() {
    let mut pair = (Foo, Foo);

    {
        let _pin = &pin mut pair.0;
    }

    let _other = &mut pair.1;
    let _moved = pair.0;
    //~^ ERROR cannot move out of `pair.0` because it is pinned
}

fn main() {}
