fn main() {
    fn f() { }
    fn g(i: int) { }
    let x = f == g;
    //~^ ERROR expected `fn()` but found `fn(int)`
}
