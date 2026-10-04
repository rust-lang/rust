//@ edition: 2024
#![allow(incomplete_features)]
#![feature(async_iterator, gen_blocks, move_expr)]

fn main() {
    let _ = || move(move(0));
    //~^ ERROR nested `move(expr)` requires another enclosing closure

    let _ = async || move(move(0));
    //~^ ERROR nested `move(expr)` requires another enclosing closure

    let _ = async { move(move(0)) };
    //~^ ERROR nested `move(expr)` requires another enclosing closure

    let _ = gen { move(move(0)) };
    //~^ ERROR nested `move(expr)` requires another enclosing closure

    let _ = async gen { move(move(0)) };
    //~^ ERROR nested `move(expr)` requires another enclosing closure
}
