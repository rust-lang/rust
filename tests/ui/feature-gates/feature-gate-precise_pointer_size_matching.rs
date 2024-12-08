fn main() {
    match 0usize {
        //~^ ERROR non-exhaustive patterns: `usize::MAX..` not covered
        //~| NOTE pattern `usize::MAX..` not covered
        //~| NOTE the matched value is of type `usize`
        //~| NOTE `usize` does not have a fixed maximum value
        0..=usize::MAX => {}
    }

    match 0isize {
        //~^ ERROR non-exhaustive patterns: `..isize::MIN` and `isize::MAX..` not covered
        //~| NOTE patterns `..isize::MIN` and `isize::MAX..` not covered
        //~| NOTE the matched value is of type `isize`
        //~| NOTE `isize` does not have fixed minimum and maximum values
        isize::MIN..=isize::MAX => {}
    }
}
