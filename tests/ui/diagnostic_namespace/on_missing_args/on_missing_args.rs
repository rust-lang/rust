#![feature(diagnostic_on_missing_args)]

#[diagnostic::on_missing_args(
    note = "this macro expects a type and a value, like `pair!(u8, 0)`",
    note = "make sure to pass both arguments",
)]
macro_rules! pair {
    //~^ NOTE when calling this macro
    ($ty:ty, $value:expr) => {};
    //~^ NOTE while trying to match `,`
}

fn main() {
    pair!(u8);
    //~^ ERROR unexpected end of macro invocation
    //~| NOTE missing tokens in macro arguments
    //~| NOTE this macro expects a type and a value, like `pair!(u8, 0)`
    //~| NOTE make sure to pass both arguments
}
