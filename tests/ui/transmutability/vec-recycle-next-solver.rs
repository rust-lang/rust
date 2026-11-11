//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ [next] compile-flags: -Znext-solver
//@ check-pass

#![feature(vec_recycle, transmutability)]

fn main() {
    let a: Vec<u8> = vec![0; 100];
    let capacity = a.capacity();
    let addr = a.as_ptr().addr();
    let b: Vec<i8> = a.recycle();
    assert_eq!(b.len(), 0);
    assert_eq!(b.capacity(), capacity);
    assert_eq!(b.as_ptr().addr(), addr);
}
