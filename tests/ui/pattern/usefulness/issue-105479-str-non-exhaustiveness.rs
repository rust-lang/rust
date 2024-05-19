fn main() {
    let a = "";
    let b = "";
    match (a, b) {
        //~^ ERROR non-exhaustive patterns: `(&_, _)` not covered [E0004]
        //~| NOTE pattern `(&_, _)` not covered
        //~| NOTE the matched value is of type `(&str, &str)`
        //~| NOTE `&str` cannot be matched exhaustively, so a wildcard `_` is necessary
        ("a", "b") => {}
        ("c", "d") => {}
    }
}
