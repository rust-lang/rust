//@ run-rustfix

#![feature(coroutines, stmt_expr_attributes)]

fn main() {
    let _ = || yield;
    //~^ ERROR `yield` can only be used
}
