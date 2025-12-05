//@ edition: 2024
// Parenthesised let "expressions" are not allowed in guards

#![feature(if_let_guard)]

#[cfg(false)]
fn un_cfged() {
    match () {
        () if let 0 = 1 => {}
        () if (let 0 = 1) => {}
        //~^ ERROR expected expression, found `let` statement
        () if (((let 0 = 1))) => {}
        //~^ ERROR expected expression, found `let` statement
    }
}

fn main() {
    match () {
        () if let 0 = 1 => {}
        () if (let 0 = 1) => {}
        //~^ ERROR expected expression, found `let` statement
        () if (((let 0 = 1))) => {}
        //~^ ERROR expected expression, found `let` statement
    }
}
