//@ run-pass
#![allow(incomplete_features)]
#![feature(move_expr)]

use std::sync::Arc;

fn main() {
    let v = Arc::new("Hello, Ferris".to_string());
    let outer = || || (move(move(v.clone()))).len();

    assert_eq!(Arc::strong_count(&v), 2);
    let inner = outer();
    assert_eq!(Arc::strong_count(&v), 2);
    assert_eq!(inner(), v.len());
    assert_eq!(Arc::strong_count(&v), 1);

    println!("{v}");
}
