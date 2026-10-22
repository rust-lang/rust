//@ check-fail
// Regression test for https://github.com/rust-lang/rust/issues/156509

#![allow(incomplete_features)]
#![feature(move_expr)]

fn main() {
    let _c = || {
        let x;
        move(x);
        //~^ ERROR cannot use `x` in this move expression
    };
}
