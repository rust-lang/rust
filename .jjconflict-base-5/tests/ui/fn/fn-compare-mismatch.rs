fn main() {
    fn f() { }
    fn g() { }
    let x = f == g;
    //~^ ERROR binary operation `==` cannot be applied
    //~| ERROR mismatched types
}
