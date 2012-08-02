fn f() -> ! {
    return 42i; //~ ERROR expected `_|_` but found `int`
    fail; //~ WARNING unreachable statement
}
fn main() { }
