#![allow(incomplete_features)]
#![feature(move_expr)]

fn main() {
    let x: bool = true;
    let y: bool = true;
    let _ = move(x) || y;
    //~^ ERROR `move(expr)` is only supported in plain closures

    let x: bool = true;
    let y: bool = true;
    let _ = move[x] || y;
    //~^ ERROR expected one of
}
