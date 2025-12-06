//@ revisions: next old
//@ edition: 2024
//@[next] compile-flags: -Znext-solver
//@ run-pass
#![feature(gen_blocks)]

fn foo() -> impl Iterator<Item = u32> {
    gen { 42.yield; for x in 3..6 { x.yield } }
}

fn moved() -> impl Iterator<Item = u32> {
    let mut x = "foo".to_string();
    gen move {
        42.yield;
        if x == "foo" { return }
        x.clear();
        for x in 3..6 { x.yield }
    }
}

fn main() {
    let mut iter = foo();
    assert_eq!(iter.next(), Some(42));
    assert_eq!(iter.next(), Some(3));
    assert_eq!(iter.next(), Some(4));
    assert_eq!(iter.next(), Some(5));
    assert_eq!(iter.next(), None);
    // `gen` blocks are fused
    assert_eq!(iter.next(), None);

    let mut iter = moved();
    assert_eq!(iter.next(), Some(42));
    assert_eq!(iter.next(), None);

}
