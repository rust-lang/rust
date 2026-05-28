//@ edition: 2024
//@ run-pass
#![feature(gen_blocks)]

// make sure that a ridiculously simple gen fn works as an iterator.

gen fn foo() -> i32 {
    yield 1;
    yield 2;
    yield 3;
}

fn main() {
    let mut iter = foo();
    assert_eq!(iter.next(), Some(1));
    assert_eq!(iter.next(), Some(2));
    assert_eq!(iter.next(), Some(3));
    assert_eq!(iter.next(), None);
}
