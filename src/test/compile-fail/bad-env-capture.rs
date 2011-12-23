// error-pattern: attempted dynamic environment-capture
fn foo() {
    let x: int;
    fn bar() { log(debug, x); }
}
fn main() { foo(); }
