#![allow(unused_mut)]

// This should type-check even though the type of `e` is not fully
// resolved when we finish type-checking the `||`.

struct Refs { refs: Vec<isize> , n: isize }

pub fn main() {
    let mut e = Refs{refs: vec![], n: 0};
    let _f = || println!("{}", e.n);
    let x: &[isize] = &e.refs;
    assert_eq!(x.len(), 0);
}
