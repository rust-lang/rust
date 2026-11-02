//@ check-fail
#![allow(incomplete_features)]
#![feature(move_expr)]

fn main() {
    let _c = || {
        let x = 3;
        move(x);
        //~^ ERROR cannot use `x` in this move expression
    };
}
