// This tests that the lint message explains the reason for the error.
fn main() {
    match 0usize {
        //~^ ERROR non-exhaustive patterns: `_` not covered
        //~| NOTE pattern `_` not covered
        //~| NOTE the matched value is of type `usize`
        //~| NOTE `usize` does not have a fixed maximum value
        0..=usize::MAX => {}
    }

    match 0isize {
        //~^ ERROR non-exhaustive patterns: `_` not covered
        //~| NOTE pattern `_` not covered
        //~| NOTE the matched value is of type `isize`
        //~| NOTE `isize` does not have a fixed maximum value
        isize::MIN..=isize::MAX => {}
    }
}
