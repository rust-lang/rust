//@ run-pass
use std::cmp::Ordering::{Less,Equal,Greater};

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct A<'a> {
    x: &'a isize
}
pub fn main() {
    let (a, b) = (A { x: &1 }, A { x: &2 });

    assert_eq!(a.cmp(&a), Equal);
    assert_eq!(b.cmp(&b), Equal);

    assert_eq!(a.cmp(&b), Less);
    assert_eq!(b.cmp(&a), Greater);
}
