// Expression macros can't expand to a let match guard.

//@revisions: edition2021 edition2024
//@[edition2021] edition:2021
//@[edition2024] edition:2024

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
