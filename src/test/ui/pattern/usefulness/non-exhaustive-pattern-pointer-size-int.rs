use std::{usize, isize};

fn main() {
    match 0usize {
        //~^ ERROR non-exhaustive patterns
        //~| NOTE pattern `_` not covered
        //~| NOTE the matched value is of type `usize`
        //~| NOTE `usize` does not have a fixed maximum value
        0 ..= usize::MAX => {}
    }

    match 0isize {
        //~^ ERROR non-exhaustive patterns
        //~| NOTE pattern `_` not covered
        //~| NOTE the matched value is of type `isize`
        //~| NOTE `isize` does not have a fixed maximum value
        isize::MIN ..= isize::MAX => {}
    }

    match 7usize {}
    //~^ ERROR non-exhaustive patterns
    //~| NOTE the matched value is of type `usize`
}
