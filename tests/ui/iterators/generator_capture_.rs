//@ run-pass

#![feature(iter_macro)]
// FIXME(iter_macro): make `yield` within it legal
#![feature(coroutines)]

use std::iter::iter;

fn main() {
    let f = {
        let s = "foo".to_string();
        iter! { move || {
            for c in s.chars() {
                yield c;
            }
        }}
    };
    let mut i = f();
    assert_eq!(i.next(), Some('f'));
    assert_eq!(i.next(), Some('o'));
    assert_eq!(i.next(), Some('o'));
    assert_eq!(i.next(), None);
    let mut i = f();
    assert_eq!(i.next(), Some('f'));
    assert_eq!(i.next(), Some('o'));
    assert_eq!(i.next(), Some('o'));
    assert_eq!(i.next(), None);
}
