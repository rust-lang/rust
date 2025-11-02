//@ run-pass

#![feature(iter_macro, yield_expr)]

use std::iter::iter;

fn main() {
    let i = {
        let s = String::new();
        iter! { move || {
            s.len().yield;
            for x in 5..10 {
                (x * 2).yield;
            }
        }}
    };
    test_iterator(i);
}

/// Exercise the iterator in a separate function to ensure it's not capturing anything it shoudln't.
fn test_iterator<I: Iterator<Item = usize>>(i: impl FnOnce() -> I) {
    let mut i = i();
    assert_eq!(i.next(), Some(0));
    assert_eq!(i.next(), Some(10));
    assert_eq!(i.next(), Some(12));
    assert_eq!(i.next(), Some(14));
    assert_eq!(i.next(), Some(16));
    assert_eq!(i.next(), Some(18));
    assert_eq!(i.next(), None);
    assert_eq!(i.next(), None);
    assert_eq!(i.next(), None);
}
