// Expression macros can't expand to a let match guard.

#![feature(if_let_guard)]

macro_rules! m {
    ($e:expr) => { let Some(x) = $e }
    //~^ ERROR expected expression, found `let` statement
}

fn main() {
    match () {
        () if m!(Some(5)) => {}
        _ => {}
    }
}
