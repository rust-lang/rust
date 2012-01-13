// xfail-test
// error-pattern: attempted dynamic environment-capture
fn foo<T>() {
    fn bar(b: T) { }
}
fn main() { }
