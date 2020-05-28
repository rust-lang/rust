// run-fail
// error-pattern:explicit panic
// ignore-emscripten no processes

fn foo<T>(t: T) {}
fn main() {
    foo(panic!())
}
