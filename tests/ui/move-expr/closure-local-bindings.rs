//@ check-fail
//@ edition: 2024

#![allow(incomplete_features)]
#![feature(gen_blocks, move_expr)]

macro_rules! move_value {
    ($value:expr) => { move($value) };
}

fn main() {
    let _ = |x: i32| move(x);
    //~^ ERROR cannot use `x` in this move expression

    let _ = |(x, _): (i32, i32)| move(x);
    //~^ ERROR cannot use `x` in this move expression

    let x = 1;
    let _ = || {
        let x = 2;
        move(x);
        //~^ ERROR cannot use `x` in this move expression
    };

    let _ = || {
        for x in [1] {
            move(x);
            //~^ ERROR cannot use `x` in this move expression
        }
    };

    let _ = || {
        if let Some(x) = Some(1) {
            move(x);
            //~^ ERROR cannot use `x` in this move expression
        }
    };

    let _ = || {
        let x = 1;
        || move(move(x))
        //~^ ERROR cannot use `x` in this move expression
    };

    let _ = || || move({
        let x = 1;
        move(x)
        //~^ ERROR cannot use `x` in this move expression
    });

    let _ = || {
        let x = 1;
        move_value!(x);
        //~^ ERROR cannot use `x` in this move expression
    };

    let _ = async |x: i32| move(x);
    //~^ ERROR cannot use `x` in this move expression

    let _ = async {
        let x = 1;
        move(x);
        //~^ ERROR cannot use `x` in this move expression
    };

    let _ = gen {
        let x = 1;
        yield move(x);
        //~^ ERROR cannot use `x` in this move expression
    };

    let _ = || {
        let x = 1;
        move(|| x)
        //~^ ERROR cannot use `x` in this move expression
    };
}
