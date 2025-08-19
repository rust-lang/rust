// gate-test-if_let_guard

use std::ops::Range;

fn _if_let_guard() {
    match () {
        () if let 0 = 1 => {}
        //~^ ERROR `if let` guards are experimental

        () if (let 0 = 1) => {}
        //~^ ERROR expected expression, found `let` statement

        () if (((let 0 = 1))) => {}
        //~^ ERROR expected expression, found `let` statement

        () if true && let 0 = 1 => {}
        //~^ ERROR `if let` guards are experimental

        () if let 0 = 1 && true => {}
        //~^ ERROR `if let` guards are experimental

        () if (let 0 = 1) && true => {}
        //~^ ERROR expected expression, found `let` statement

        () if true && (let 0 = 1) => {}
        //~^ ERROR expected expression, found `let` statement

        () if (let 0 = 1) && (let 0 = 1) => {}
        //~^ ERROR expected expression, found `let` statement
        //~| ERROR expected expression, found `let` statement

        () if let 0 = 1 && let 1 = 2 && (let 2 = 3 && let 3 = 4 && let 4 = 5) => {}
        //~^ ERROR `if let` guards are experimental
        //~| ERROR expected expression, found `let` statement
        //~| ERROR expected expression, found `let` statement
        //~| ERROR expected expression, found `let` statement


        () if let Range { start: _, end: _ } = (true..true) && false => {}
        //~^ ERROR `if let` guards are experimental

        _ => {}
    }
}

fn _macros() {
    macro_rules! use_expr {
        ($e:expr) => {
            match () {
                () if $e => {}
                _ => {}
            }
        }
    }
    use_expr!((let 0 = 1 && 0 == 0));
    //~^ ERROR expected expression, found `let` statement
    //~| ERROR expected expression, found `let` statement
    use_expr!((let 0 = 1));
    //~^ ERROR expected expression, found `let` statement
    //~| ERROR expected expression, found `let` statement
    match () {
        #[cfg(false)]
        () if let 0 = 1 => {}
        //~^ ERROR `if let` guards are experimental
        _ => {}
    }
}

fn main() {}
