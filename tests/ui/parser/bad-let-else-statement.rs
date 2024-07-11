#![feature(yeet_expr)]
#![allow(incomplete_features)] // Necessary for now, while explicit_tail_calls is incomplete
#![feature(explicit_tail_calls)]

fn a() {
    let 0 = {
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
    let 0 = if true {
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
    let 0 = match true {
        true => 1,
        false => 0
    } else {
        //~^ ERROR right curly brace `}` before `else` in a `let...else` statement not allowed
        return;
    };
}

fn f() {
    struct X {
        a: i32,
    }

    let X { a: 0 } = X {
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
    let 0 = const {
        1
    } else {
        //~^ ERROR right curly brace `}` before `else` in a `let...else` statement not allowed
        return;
    };
}

fn i() {
    let 0 = &{
        1
    } else {
        //~^ ERROR right curly brace `}` before `else` in a `let...else` statement not allowed
        return;
    };
}

fn j() {
    let mut bar = 0;
    let foo = bar = {
        //~^ WARN irrefutable `let...else` pattern
        1
    } else {
        //~^ ERROR right curly brace `}` before `else` in a `let...else` statement not allowed
        return;
    };
}

fn k() {
    let 0 = 1 + {
        1
    } else {
        //~^ ERROR right curly brace `}` before `else` in a `let...else` statement not allowed
        return;
    };
}

fn l() {
    const RANGE: std::ops::Range<u8> = 0..0;
    let RANGE = 1..{
        1
    } else {
        //~^ ERROR right curly brace `}` before `else` in a `let...else` statement not allowed
        return;
    };
}

fn m() {
    let 0 = return {
        ()
    } else {
        //~^ ERROR right curly brace `}` before `else` in a `let...else` statement not allowed
        return;
    };
}

fn n() {
    let 0 = -{
        1
    } else {
        //~^ ERROR right curly brace `}` before `else` in a `let...else` statement not allowed
        return;
    };
}

fn o() -> Result<(), ()> {
    let 0 = do yeet {
        ()
    } else {
        //~^ ERROR right curly brace `}` before `else` in a `let...else` statement not allowed
        return Ok(());
    };
}

// fn p() { // FIXME(explicit_tail_calls): this currently trips an assertion...
//     let 0 = become {
//         ()
//     } else {
//         // ~^ ERROR right curly brace `}` before `else` in a `let...else` statement not allowed
//         return;
//     };
// }

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
        () => {
            { 1 }
        };
    }

    macro_rules! b {
        (1) => {
            let 0 = a!() else { return; };
        };
        (2) => {
            let 0 = a! {} else { return; };
            //~^ ERROR right curly brace `}` before `else` in a `let...else` statement not allowed
        };
    }

    b!(1);
    b!(2);
}

fn t() {
    macro_rules! primitive {
        (8) => { u8 };
    }

    let foo = &std::ptr::null as &'static dyn std::ops::Fn() -> *const primitive! {
        //~^ WARN irrefutable `let...else` pattern
        8
    } else {
        //~^ ERROR right curly brace `}` before `else` in a `let...else` statement not allowed
        return;
    };
}

fn main() {}
