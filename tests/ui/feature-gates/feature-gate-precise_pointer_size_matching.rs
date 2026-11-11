fn main() {
    match 0usize {
        //~^ ERROR non-exhaustive patterns: `usize::MAX..` not covered
        //~| NOTE pattern `usize::MAX..` not covered
        //~| NOTE the matched value is of type `usize`
        //~| NOTE `usize::MAX` is not treated as exhaustive, so half-open ranges are necessary to match exhaustively
        0..=usize::MAX => {}
    }

    match 0isize {
        //~^ ERROR non-exhaustive patterns: `..isize::MIN` and `isize::MAX..` not covered
        //~| NOTE patterns `..isize::MIN` and `isize::MAX..` not covered
        //~| NOTE the matched value is of type `isize`
        //~| NOTE `isize::MIN` and `isize::MAX` are not treated as exhaustive, so half-open ranges are necessary to match exhaustively
        isize::MIN..=isize::MAX => {}
    }
}
