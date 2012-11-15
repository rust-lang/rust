fn main() {
    fn f() { }
    fn g() { }
    let x = f == g;
    //~^ ERROR mismatched types
    //~^^ ERROR failed to find an implementation of trait
}
