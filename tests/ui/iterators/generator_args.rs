//@ run-pass

#![feature(iter_macro, yield_expr)]

use std::iter::iter;

fn main() {
    let i = iter! {|foo| {
        foo.yield;
        for x in 5..10 {
            (x * 2).yield;
        }
    }};
    let mut i = i(3);
    assert_eq!(i.next(), Some(3));
    assert_eq!(i.next(), Some(10));
    assert_eq!(i.next(), Some(12));
    assert_eq!(i.next(), Some(14));
    assert_eq!(i.next(), Some(16));
    assert_eq!(i.next(), Some(18));
    assert_eq!(i.next(), None);
    assert_eq!(i.next(), None);
    assert_eq!(i.next(), None);
}
