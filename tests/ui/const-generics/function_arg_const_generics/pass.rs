//@ run-pass
//@ edition: 2021
#![feature(function_arg_const_generics, gca_min_const_items)]

fn foo(const N: usize) -> [u8; N] {
    [0; N]
}

fn forward<const K: usize>() -> [u8; K] {
    foo(K)
}

fn in_signature(const N: usize, x: [u8; N]) -> [u8; N] {
    let y: [u8; N] = x;
    y
}

fn apit_before(x: impl Copy, const N: usize) -> [u8; N] {
    let _ = x;
    [0; N]
}

fn apit_after(const N: usize, x: impl Copy) -> [u8; N] {
    let _ = x;
    [0; N]
}

async fn in_async(const N: usize) -> [u8; N] {
    [0; N]
}

struct S;

impl S {
    fn method(&self, x: u8, const N: usize) -> [u8; N] {
        [x; N]
    }
}

trait Tr {
    fn assoc(const N: usize) -> [u8; N];
}

impl Tr for S {
    fn assoc(const N: usize) -> [u8; N] {
        [1; N]
    }
}

fn main() {
    assert_eq!(foo(3).len(), 3);
    assert_eq!(forward::<5>().len(), 5);
    assert_eq!(in_signature(2, [4; 2]), [4, 4]);
    assert_eq!(apit_before(1u8, 2).len(), 2);
    assert_eq!(apit_after(2, 1u8).len(), 2);
    let _ = in_async(3);
    assert_eq!(S.method(7, 2), [7, 7]);
    assert_eq!(S::method(&S, 9, 3), [9, 9, 9]);
    assert_eq!(<S as Tr>::assoc(2), [1, 1]);
    assert_eq!(S::assoc(2), [1, 1]);
}
