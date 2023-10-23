// revisions: next old
//compile-flags: --edition 2024 -Zunstable-options
//[next] compile-flags: -Ztrait-solver=next
// run-pass
#![feature(gen_blocks)]

fn foo() -> impl Iterator<Item = u32> {
    gen { yield 42; for x in 3..6 { yield x } }
}

fn main() {
    let mut iter = foo();
    assert_eq!(iter.next(), Some(42));
    assert_eq!(iter.next(), Some(3));
    assert_eq!(iter.next(), Some(4));
    assert_eq!(iter.next(), Some(5));
    assert_eq!(iter.next(), None);
}
