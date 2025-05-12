//@ run-pass
#![allow(unused_variables)]
#![allow(dead_code)]

macro_rules! piece(
    ($piece:pat) => ($piece);
);

enum Piece {A, B}

fn main() {
    match Piece::A {
        piece!(pt@ Piece::A) | piece!(pt@ Piece::B) => {}
    }
}
