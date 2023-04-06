fn main() {
    let A = 3;
    //~^ ERROR refutable pattern in local binding
    //~| patterns `i32::MIN..=1_i32` and `3_i32..=i32::MAX` not covered
    //~| missing patterns are not covered because `A` is interpreted as a constant pattern, not a new variable
    //~| HELP introduce a variable instead
    //~| SUGGESTION A_var

    const A: i32 = 2;
}
