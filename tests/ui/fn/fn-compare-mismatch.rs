fn main() {
    fn f() { }
    fn g() { }
    let x = f == g;
    //~^ ERROR: can't compare `fn() {f}` to `fn() {g}`
    //~| ERROR: mismatched types
}
