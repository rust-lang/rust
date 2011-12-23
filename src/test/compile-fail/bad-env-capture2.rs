// error-pattern: attempted dynamic environment-capture
fn foo(x: int) {
    fn bar() { log(debug, x); }
}
fn main() { foo(2); }
