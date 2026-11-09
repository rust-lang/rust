//@ aux-crate:other=crate_with_non_move_type.rs
// `other` is a crate where `#![feature(move_trait)]` is enabled.
// test what happens when importing it in a crate that assumes `Move` by default.
//
// FIXME: this should all raise errors but it doesn't because in this crate we assume everything is
// move. At the very least we should disallow importing `other` to be sound.
// v add check pass, but this test should really fail.
//@ check-pass

extern crate other;

fn foo(f: other::NotMove) {}

fn main() {
    let x = other::NotMove;
    let y = x;
    //^ FIXME this should raise an error

    let x = other::NotMove;
    foo(x);
    //^ FIXME this should raise an error

    let x = other::NotMove;
    other::requires_move(x);
    //^ FIXME this should raise an error
}
