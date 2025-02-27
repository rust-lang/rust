//@ run-pass

#![feature(iter_macro)]
// FIXME(iter_macro): make `yield` within it legal
#![feature(coroutines)]

use std::iter::iter;

fn main() {
    let i = iter! {
        yield 0;
        for x in 5..10 {
            yield x * 2;
        }
    };
    let mut i = i();
    assert_eq!(i.next(), Some(0));
    assert_eq!(i.next(), Some(10));
    assert_eq!(i.next(), Some(12));
    assert_eq!(i.next(), Some(14));
    assert_eq!(i.next(), Some(16));
    assert_eq!(i.next(), Some(18));
    assert_eq!(i.next(), None);
    // FIXME(iter_macro): desugar to `gen` instead of coroutines,
    // as the latter panic after resuming iteration.
    //assert_eq!(i.next(), None);
    //assert_eq!(i.next(), None);
}
