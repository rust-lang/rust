#![feature(diagnostic_on_missing_args)]

#[diagnostic::on_missing_args(
    message = "{This}! is missing its second argument",
    label = "add the missing value here",
    note = "this macro expects a type and a value, like `pair!(u8, 0)`",
)]
macro_rules! pair {
    //~^ NOTE when calling this macro
    ($ty:ty, $value:expr) => {};
    //~^ NOTE while trying to match `,`
}

fn main() {
    pair!(u8);
    //~^ ERROR pair! is missing its second argument
    //~| NOTE add the missing value here
    //~| NOTE this macro expects a type and a value, like `pair!(u8, 0)`
}
