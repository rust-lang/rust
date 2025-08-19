// This test exercises lending behavior for iterator closures which is not yet supported.

#![feature(iter_macro, yield_expr)]

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
    let mut i = f(); //~ ERROR use of moved value: `f`
    assert_eq!(i.next(), Some('f'));
    assert_eq!(i.next(), Some('o'));
    assert_eq!(i.next(), Some('o'));
    assert_eq!(i.next(), None);
}
