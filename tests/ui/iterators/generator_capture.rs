//@ run-pass

#![feature(iter_macro, yield_expr)]

use std::iter::iter;

fn main() {
    let i = {
        let s = String::new();
        iter! { move || {
            yield s.len();
            for x in 5..10 {
                yield x * 2;
            }
        }}
    };
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
