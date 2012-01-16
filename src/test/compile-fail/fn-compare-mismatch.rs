fn main() {
    fn f() { }
    fn g(i: int) { }
    let x = f == g;
    //!^ ERROR expected `native fn()` but found `native fn(int)`
}
