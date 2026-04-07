#![feature(diagnostic_on_missing_args)]

#[diagnostic::on_missing_args(
    message = "{This}! expects exactly two arguments",
    label = "unexpected extra input starts here",
    note = "this macro expects a type and a value, like `pair!(u8, 0)`",
    note = "make sure to pass both arguments",
)]
macro_rules! pair {
    //~^ NOTE when calling this macro
    ($ty:ty, $value:expr) => {};
    //~^ NOTE while trying to match meta-variable `$value:expr`
}

fn main() {
    pair!(u8, 0, 42);
    //~^ ERROR pair! expects exactly two arguments
    //~| NOTE unexpected extra input starts here
    //~| NOTE this macro expects a type and a value, like `pair!(u8, 0)`
    //~| NOTE make sure to pass both arguments
}
