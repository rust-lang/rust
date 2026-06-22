//@ edition: 2024
//@ run-pass
#![allow(incomplete_features)]
#![feature(gen_blocks, move_expr)]

use std::cell::Cell;
use std::sync::Arc;

fn main() {
    let created = Cell::new(0);
    let mut iter = gen {
        let n = move({
            created.set(created.get() + 1);
            created.get()
        });
        yield n;
        yield n + 1;
    };
    assert_eq!(created.get(), 1);
    assert_eq!(iter.next(), Some(1));
    assert_eq!(iter.next(), Some(2));
    assert_eq!(iter.next(), None);

    let x = Arc::new(String::from("hello"));
    assert_eq!(Arc::strong_count(&x), 1);
    let mut iter = gen {
        let value = move(x.clone());
        yield Arc::strong_count(&value);
        yield Arc::strong_count(&value);
    };
    assert_eq!(Arc::strong_count(&x), 2);
    assert_eq!(iter.next(), Some(2));
    assert_eq!(iter.next(), Some(2));
    drop(iter);
    assert_eq!(Arc::strong_count(&x), 1);

    let y = Arc::new(String::from("nested"));
    assert_eq!(Arc::strong_count(&y), 1);
    let mut iter = gen {
        let value = move(move(y.clone()));
        yield Arc::strong_count(&value);
    };
    assert_eq!(Arc::strong_count(&y), 2);
    assert_eq!(iter.next(), Some(2));
    drop(iter);
    assert_eq!(Arc::strong_count(&y), 1);

    let z = Arc::new(String::from("gen move"));
    assert_eq!(Arc::strong_count(&z), 1);
    let mut iter = gen move {
        let value = move(z.clone());
        yield Arc::strong_count(&value);
    };
    assert_eq!(Arc::strong_count(&z), 2);
    assert_eq!(iter.next(), Some(2));
    drop(iter);
    assert_eq!(Arc::strong_count(&z), 1);
}
