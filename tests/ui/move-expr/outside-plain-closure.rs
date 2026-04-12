#![allow(incomplete_features)]
#![feature(move_expr)]

fn main() {
    let _ = move(String::from("nope"));
    //~^ ERROR `move(expr)` is only supported in plain closures
}
