//@ aux-crate:other=crate_with_non_move_type.rs
// `other` is a crate where `#![feature(move_trait)]` is enabled.
// test what happens when importing it in a crate that assumes `Move` by default.
//
// FIXME: this should all raise errors but it doesn't because in this crate we assume everything is
// move. At the very least we should disallow importing `other` to be sound.

extern crate other;

fn foo(f: other::NotMove) {}

fn main() {
    let x = other::NotMove;
    let y = x;
    //~^ ERROR values of type `NotMove` may not be movable

    let x = other::NotMove;
    foo(x);
    //~^ ERROR values of type `NotMove` may not be movable
    //~| ERROR values of type `NotMove` may not be movable
    //^ FIXME this should raise one error.
}

// Unfortunately the errors in main don't trigger if there's a
// non-borrowck trait bound error.
fn other_func() {
    let x = other::NotMove;
    other::requires_move(x);
    //~^ ERROR values of type `NotMove` may not be movable
}
