//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] known-bug: trait-system-refactor-initiative#302
//@[next] check-pass

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
    let mut i = f(); //[current]~ ERROR use of moved value: `f`
    assert_eq!(i.next(), Some('f'));
    assert_eq!(i.next(), Some('o'));
    assert_eq!(i.next(), Some('o'));
    assert_eq!(i.next(), None);
}
