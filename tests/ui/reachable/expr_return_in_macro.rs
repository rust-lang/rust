// Tests that we generate nice error messages
// when an expression is unreachble due to control
// flow inside of a macro expansion.
#![deny(unreachable_code)]

macro_rules! early_return {
    () => {
        return ()
    }
}

fn main() {
    return early_return!();
    //~^ ERROR unreachable expression
}
