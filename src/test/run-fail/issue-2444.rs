// error-pattern:explicit panic

use std::sync::Arc;

enum e<T> {
    ee(Arc<T>),
}

fn foo() -> e<isize> {
    panic!();
}

fn main() {
    let _f = foo();
}
