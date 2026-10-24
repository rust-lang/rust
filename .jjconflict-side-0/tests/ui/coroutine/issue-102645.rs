#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::ops::Coroutine;
use std::pin::Pin;

fn main() {
    let mut a = 5;
    let mut b = #[coroutine]
    || {
        let d = 6;
        yield;
        _zzz(); // #break
        a = d;
    };
    Pin::new(&mut b).resume();
    //~^ ERROR this method takes 1 argument but 0 arguments were supplied
    // This type error is required to reproduce the ICE...
}

fn _zzz() {
    ()
}
