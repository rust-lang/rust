// run-fail
// error-pattern:explicit panic
// ignore-emscripten no processes

use std::sync::Arc;

enum Err<T> {
    Errr(Arc<T>),
}

fn foo() -> Err<isize> {
    panic!();
}

fn main() {
    let _f = foo();
}
