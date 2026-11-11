fn main() {
    let mut x = 1i32;
    let y = Box::new(|| 1);
    x = y;
    //~^ ERROR mismatched types
    //~| HELP use parentheses to call this closure
}
