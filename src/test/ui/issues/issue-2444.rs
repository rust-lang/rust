// run-fail
// error-pattern:explicit panic

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
