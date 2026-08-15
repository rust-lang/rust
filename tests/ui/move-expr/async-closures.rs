//@ edition: 2021
#![allow(incomplete_features)]
#![feature(move_expr)]

fn main() {
    let s = String::from("hello");
    let _ = async || {
        move(s);
        //~^ ERROR `move(expr)` is only supported in plain closures
    };
}
