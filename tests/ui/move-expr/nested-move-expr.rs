//@ run-pass
// This needs to remain a run-pass test to check capture behavior at runtime.
#![allow(incomplete_features)]
#![deny(unused_mut)]
#![feature(move_expr)]

use std::sync::Arc;

fn main() {
    let v = Arc::new("Hello, Ferris".to_string());
    let outer = || || (move(move(v.clone()))).len();

    assert_eq!(Arc::strong_count(&v), 2);
    let inner = outer();
    assert_eq!(Arc::strong_count(&v), 2);
    assert_eq!(inner(), v.len());
    assert_eq!(inner(), v.len());
    assert_eq!(Arc::strong_count(&v), 2);
    drop(inner);
    assert_eq!(Arc::strong_count(&v), 1);

    println!("{v}");

    let data = vec![1];
    let f = || {
        let mut inner = || {
            let data = &mut move(move(data));
            data.push(2);
            data.len()
        };
        assert_eq!(inner(), 2);
        assert_eq!(inner(), 3);
    };
    f();
}
