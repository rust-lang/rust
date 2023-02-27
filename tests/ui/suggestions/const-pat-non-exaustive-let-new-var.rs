fn main() {
    let A = 3;
    //~^ ERROR refutable pattern in local binding
    //~| patterns `i32::MIN..=1_i32` and `3_i32..=i32::MAX` not covered
    //~| HELP you might want to use `if let` to ignore the variants that aren't matched

    const A: i32 = 2;
}
