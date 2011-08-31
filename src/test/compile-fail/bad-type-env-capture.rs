// xfail-test
// error-pattern: attempted dynamic environment-capture
fn foo<T>() {
    obj bar(b: T) { }
}
fn main() { }
