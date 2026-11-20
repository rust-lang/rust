//@ check-pass
#![allow(incomplete_features)]
#![feature(move_expr)]

fn main() {
    let x: bool = true;
    let y: bool = true;
    let _ = || move(x) || y;

    let x: bool = true;
    let y: bool = true;
    let _ = move || move(x) || y;
}
