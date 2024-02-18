#![feature(inline_const)]
#![feature(yeet_expr)]
#![allow(incomplete_features)] // Necessary for now, while explicit_tail_calls is incomplete
#![feature(explicit_tail_calls)]

fn a() {
    let foo = {
        //~^ WARN irrefutable `let...else` pattern
        1
    } else {
        //~^ ERROR right curly brace `}` before `else` in a `let...else` statement not allowed
        return;
    };
}

fn b() {
    let foo = for i in 1..2 {
        break;
    } else {
        //~^ ERROR `for...else` loops are not supported
        return;
    };
}

fn c() {
    let foo = if true {
        //~^ WARN irrefutable `let...else` pattern
        1
    } else {
        0
    } else {
        //~^ ERROR right curly brace `}` before `else` in a `let...else` statement not allowed
        return;
    };
}

fn d() {
    let foo = loop {
        break;
    } else {
        //~^ ERROR loop...else` loops are not supported
        return;
    };
}

fn e() {
    let foo = match true {
        //~^ WARN irrefutable `let...else` pattern
        true => 1,
        false => 0
    } else {
        //~^ ERROR right curly brace `}` before `else` in a `let...else` statement not allowed
        return;
    };
}

struct X {a: i32}
fn f() {
    let foo = X {
        //~^ WARN irrefutable `let...else` pattern
        a: 1
    } else {
        //~^ ERROR right curly brace `}` before `else` in a `let...else` statement not allowed
        return;
    };
}

fn g() {
    let foo = while false {
        break;
    } else {
        //~^ ERROR `while...else` loops are not supported
        return;
    };
}

fn h() {
    let foo = const {
        //~^ WARN irrefutable `let...else` pattern
        1
    } else {
        //~^ ERROR right curly brace `}` before `else` in a `let...else` statement not allowed
        return;
    };
}

fn i() {
    let foo = &{
        //~^ WARN irrefutable `let...else` pattern
        1
    } else {
        //~^ ERROR right curly brace `}` before `else` in a `let...else` statement not allowed
        return;
    };
}

fn j() {
    let bar = 0;
    let foo = bar = { //~ ERROR: cannot assign twice
        //~^ WARN irrefutable `let...else` pattern
        1
    } else {
        //~^ ERROR right curly brace `}` before `else` in a `let...else` statement not allowed
        return;
    };
}

fn k() {
    let foo = 1 + {
        //~^ WARN irrefutable `let...else` pattern
        1
    } else {
        //~^ ERROR right curly brace `}` before `else` in a `let...else` statement not allowed
        return;
    };
}

fn l() {
    let foo = 1..{
        //~^ WARN irrefutable `let...else` pattern
        1
    } else {
        //~^ ERROR right curly brace `}` before `else` in a `let...else` statement not allowed
        return;
    };
}

fn m() {
    let foo = return {
        //~^ WARN irrefutable `let...else` pattern
        ()
    } else {
        //~^ ERROR right curly brace `}` before `else` in a `let...else` statement not allowed
        return;
    };
}

fn n() {
    let foo = -{
        //~^ WARN irrefutable `let...else` pattern
        1
    } else {
        //~^ ERROR right curly brace `}` before `else` in a `let...else` statement not allowed
        return;
    };
}

fn o() -> Result<(), ()> {
    let foo = do yeet {
        //~^ WARN irrefutable `let...else` pattern
        ()
    } else {
        //~^ ERROR right curly brace `}` before `else` in a `let...else` statement not allowed
        return Ok(());
    };
}

fn p() {
    let foo = become {
        //~^ WARN irrefutable `let...else` pattern
        ()
    } else {
        //~^ ERROR right curly brace `}` before `else` in a `let...else` statement not allowed
        return;
    };
}

fn q() {
    let foo = |x: i32| {
        //~^ WARN irrefutable `let...else` pattern
        x
    } else {
        //~^ ERROR right curly brace `}` before `else` in a `let...else` statement not allowed
        return;
    };
}

fn r() {
    let ok = format_args!("") else { return; };
    //~^ WARN irrefutable `let...else` pattern

    let bad = format_args! {""} else { return; };
    //~^ ERROR right curly brace `}` before `else` in a `let...else` statement not allowed
    //~| WARN irrefutable `let...else` pattern
}

fn s() {
    macro_rules! a {
        () => { {} }
        //~^ WARN irrefutable `let...else` pattern
        //~| WARN irrefutable `let...else` pattern
    }

    macro_rules! b {
        (1) => {
            let x = a!() else { return; };
        };
        (2) => {
            let x = a! {} else { return; };
            //~^ ERROR right curly brace `}` before `else` in a `let...else` statement not allowed
        };
    }

    b!(1); b!(2);
}

fn main() {}
