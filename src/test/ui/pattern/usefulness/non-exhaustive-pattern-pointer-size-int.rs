use std::{usize, isize};

fn main() {
    match 0usize {
        //~^ ERROR non-exhaustive patterns
        //~| NOTE pattern `_` not covered
        //~| NOTE the matched value is of type `usize`
        //~| NOTE for `usize` and `isize`, no assumptions about the maximum value are permitted
        //~| NOTE to exhaustively match on either pointer-size integer type, wildcards must be used
        0 ..= usize::MAX => {}
    }

    match 0isize {
        //~^ ERROR non-exhaustive patterns
        //~| NOTE pattern `_` not covered
        //~| NOTE the matched value is of type `isize`
        //~| NOTE for `usize` and `isize`, no assumptions about the maximum value are permitted
        //~| NOTE to exhaustively match on either pointer-size integer type, wildcards must be used
        isize::MIN ..= isize::MAX => {}
    }
}
