#![feature(diagnostic_on_unmatch_args)]

#[diagnostic::on_unmatch_args(
    message = "invalid arguments to {This} macro invocation",
    label = "expected a type and value here",
    note = "this macro expects a type and a value, like `pair!(u8, 0)`",
    note = "see the macro documentation for accepted forms",
)]
macro_rules! pair {
    //~^ NOTE when calling this macro
    ($ty:ty, $value:expr) => {};
    //~^ NOTE while trying to match `,`
}

fn main() {
    pair!(u8);
    //~^ ERROR invalid arguments to pair macro invocation
    //~| NOTE expected a type and value here
    //~| NOTE this macro expects a type and a value, like `pair!(u8, 0)`
    //~| NOTE see the macro documentation for accepted forms
}
