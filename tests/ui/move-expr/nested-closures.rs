//@ check-pass
#![allow(incomplete_features)]
#![feature(move_expr)]

use std::sync::Arc;

fn main() {
    let x = Arc::new(String::from("hello"));
    assert_eq!(Arc::strong_count(&x), 1);

    let outer = || {
        assert_eq!(Arc::strong_count(&x), 1);
        let inner = || move(x.clone());
        assert_eq!(Arc::strong_count(&x), 2);
        let y = inner();
        assert_eq!(&*y, "hello");
        assert_eq!(Arc::strong_count(&x), 2);
        drop(y);
        assert_eq!(Arc::strong_count(&x), 1);
    };

    assert_eq!(Arc::strong_count(&x), 1);
    // `outer` captures `x` by reference, and `inner` takes ownership of a clone.
    println!("{x}");
    outer();
    assert_eq!(Arc::strong_count(&x), 1);
}
