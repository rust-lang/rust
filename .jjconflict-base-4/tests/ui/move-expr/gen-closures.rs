//@ run-pass

#![allow(incomplete_features)]
#![feature(iter_macro, move_expr, yield_expr)]

use std::cell::Cell;
use std::iter::iter;
use std::sync::Arc;

fn main() {
    let created = Cell::new(0);
    let closure = iter! { || {
        let n = move({
            created.set(created.get() + 1);
            created.get()
        });
        yield n;
    }};
    assert_eq!(created.get(), 1);
    assert_eq!(closure().next(), Some(1));
    assert_eq!(closure().next(), Some(1));
    assert_eq!(created.get(), 1);

    let x = Arc::new(String::from("hello"));
    assert_eq!(Arc::strong_count(&x), 1);

    let closure = iter! { || {
        yield move(x.clone());
    }};
    assert_eq!(Arc::strong_count(&x), 2);
    let mut generator = closure();
    assert_eq!(Arc::strong_count(&x), 2);
    let yielded = generator.next().unwrap();
    assert_eq!(Arc::strong_count(&x), 2);
    assert_eq!(generator.next(), None);
    drop(yielded);
    assert_eq!(Arc::strong_count(&x), 1);

    let a = String::from("a");
    let b = String::from("bbb");
    let closure = iter! { || {
        let moved = move(a.clone());
        yield (moved, b.len());
    }};
    assert_eq!(closure().next(), Some((String::from("a"), 3)));
}
