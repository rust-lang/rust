// gate-test-if_let_guard

use std::ops::Range;

fn _if_let_guard() {
    match () {
        () if let 0 = 1 => {}
        //~^ ERROR `if let` guards are experimental

        () if (let 0 = 1) => {}
        //~^ ERROR `let` expressions in this position are experimental

        () if (((let 0 = 1))) => {}
        //~^ ERROR `let` expressions in this position are experimental

        () if true && let 0 = 1 => {}
        //~^ ERROR `let` expressions in this position are experimental

        () if let 0 = 1 && true => {}
        //~^ ERROR `let` expressions in this position are experimental

        () if (let 0 = 1) && true => {}
        //~^ ERROR `let` expressions in this position are experimental

        () if true && (let 0 = 1) => {}
        //~^ ERROR `let` expressions in this position are experimental

        () if (let 0 = 1) && (let 0 = 1) => {}
        //~^ ERROR `let` expressions in this position are experimental
        //~| ERROR `let` expressions in this position are experimental

        () if let 0 = 1 && let 1 = 2 && (let 2 = 3 && let 3 = 4 && let 4 = 5) => {}
        //~^ ERROR `let` expressions in this position are experimental
        //~| ERROR `let` expressions in this position are experimental
        //~| ERROR `let` expressions in this position are experimental
        //~| ERROR `let` expressions in this position are experimental
        //~| ERROR `let` expressions in this position are experimental

        () if let Range { start: _, end: _ } = (true..true) && false => {}
        //~^ ERROR `let` expressions in this position are experimental
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
    //~^ ERROR `let` expressions in this position are experimental
    use_expr!((let 0 = 1));
    //~^ ERROR `let` expressions in this position are experimental
    match () {
        #[cfg(FALSE)]
        () if let 0 = 1 => {}
        //~^ ERROR `if let` guards are experimental
        _ => {}
    }
    use_expr!(let 0 = 1);
    //~^ ERROR no rules expected the token `let`
}

fn main() {}
