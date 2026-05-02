#![feature(pin_ergonomics)]
#![allow(incomplete_features)]

// This protects pinning projected places. Pinning `pair.0` must not pin
// unrelated disjoint fields, but it must reject later mutable borrows or moves
// of `pair.0` until reassignment.

#[pin_v2]
#[derive(Default)]
struct Foo(std::marker::PhantomPinned);

fn mutable_borrow_of_pinned_projection() {
    let mut pair = (Foo::default(), Foo::default());

    {
        let _pin = &pin mut pair.0;
    }

    let _other = &mut pair.1;
    let _borrow = &mut pair.0;
    //~^ ERROR cannot borrow `pair.0` as mutable because it is pinned
}

fn move_of_pinned_projection() {
    let mut pair = (Foo::default(), Foo::default());

    {
        let _pin = &pin mut pair.0;
    }

    let _other = &mut pair.1;
    let _moved = pair.0;
    //~^ ERROR cannot move out of `pair.0` because it is pinned
}

#[pin_v2]
struct ContainsUnpinField {
    field: String,
    _pin: std::marker::PhantomPinned,
}

fn pinned_parent_still_blocks_unpin_field_move(mut value: ContainsUnpinField) {
    {
        let _ = &pin mut value;
    }

    let _moved = value.field;
    //~^ ERROR cannot move out of `value.field` because it is pinned
}

fn pinned_parent_still_blocks_unpin_field_mut_borrow(mut value: ContainsUnpinField) {
    {
        let _ = &pin mut value;
    }

    let _ = &mut value.field;
    //~^ ERROR cannot borrow `value.field` as mutable because it is pinned
}

fn main() {}
