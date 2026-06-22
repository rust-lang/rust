//@ edition: 2021
//@ run-pass
#![allow(incomplete_features)]
#![feature(move_expr)]

use std::cell::Cell;
use std::sync::Arc;

fn main() {
    let created = Cell::new(0);
    let fut = async {
        let n = move({
            created.set(created.get() + 1);
            created.get()
        });
        n
    };
    assert_eq!(created.get(), 1);
    drop(fut);

    let x = Arc::new(String::from("hello"));
    assert_eq!(Arc::strong_count(&x), 1);
    let fut = async { move(x.clone()) };
    assert_eq!(Arc::strong_count(&x), 2);
    drop(fut);
    assert_eq!(Arc::strong_count(&x), 1);

    let y = Arc::new(String::from("nested"));
    assert_eq!(Arc::strong_count(&y), 1);
    let fut = async { move(move(y.clone())) };
    assert_eq!(Arc::strong_count(&y), 2);
    drop(fut);
    assert_eq!(Arc::strong_count(&y), 1);

    let z = Arc::new(String::from("async move"));
    assert_eq!(Arc::strong_count(&z), 1);
    let fut = async move { move(z.clone()) };
    assert_eq!(Arc::strong_count(&z), 2);
    drop(fut);
    assert_eq!(Arc::strong_count(&z), 1);
}
