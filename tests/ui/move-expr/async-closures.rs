//@ edition: 2021
//@ run-pass
#![allow(incomplete_features)]
#![feature(move_expr)]

use std::cell::Cell;
use std::sync::Arc;

fn main() {
    let created = Cell::new(0);
    let c = async || {
        let n = move({
            created.set(created.get() + 1);
            created.get()
        });
        n
    };
    assert_eq!(created.get(), 0);
    let fut = c();
    assert_eq!(created.get(), 1);
    drop(fut);

    let x = Arc::new(String::from("hello"));
    assert_eq!(Arc::strong_count(&x), 1);

    let c = async || move(x.clone());
    assert_eq!(Arc::strong_count(&x), 1);
    let fut = c();
    assert_eq!(Arc::strong_count(&x), 2);
    drop(fut);
    assert_eq!(Arc::strong_count(&x), 1);
    assert_eq!(Arc::strong_count(&x), 1);
}
