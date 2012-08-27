fn main() {
    fn f() { }
    fn g(i: int) { }
    let x = f == g;
    //~^ ERROR binary operation == cannot be applied to type
}
