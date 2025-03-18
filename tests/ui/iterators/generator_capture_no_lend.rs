//@ run-pass

#![feature(iter_macro, yield_expr)]

// This test creates an iterator that captures a reference and ensure that doesn't force the
// iterator to become lending.

use std::iter::iter;

fn main() {
    let s = "foo".to_string();
    let f = iter! { || {
        for c in s.chars() {
            yield c;
        }
    }};

    let mut i = f();
    let mut j = f();

    assert_eq!(i.next(), Some('f'));
    assert_eq!(i.next(), Some('o'));
    assert_eq!(i.next(), Some('o'));
    assert_eq!(i.next(), None);

    assert_eq!(j.next(), Some('f'));
    assert_eq!(j.next(), Some('o'));
    assert_eq!(j.next(), Some('o'));
    assert_eq!(j.next(), None);
}
