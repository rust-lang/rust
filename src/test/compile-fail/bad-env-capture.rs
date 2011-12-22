// error-pattern: attempted dynamic environment-capture
fn foo() {
    let x: int;
    fn bar() { log_full(core::debug, x); }
}
fn main() { foo(); }
