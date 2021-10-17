fn main() {
    fn f() { }
    fn g() { }
    let x = f == g;
    //~^ ERROR can't compare
    //~| ERROR mismatched types
}
