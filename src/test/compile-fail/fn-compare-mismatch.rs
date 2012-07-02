fn main() {
    fn f() { }
    fn g(i: int) { }
    let x = f == g;
    //~^ ERROR expected `extern fn()` but found `extern fn(int)`
}
