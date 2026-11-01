// Issue 36278: On an unsized struct with >1 level of nontrivial
// nesting, ensure we are computing dynamic size of prefix correctly.

use std::mem;

const SZ: usize = 100;
struct P<T: ?Sized>(#[allow(dead_code)] [u8; SZ], T);

type Ack<T> = P<P<T>>;

fn main() {
    let size_of_sized;
    let size_of_unsized;
    let x: Box<Ack<[u8; 0]>> = Box::new(P([0; SZ], P([0; SZ], [0; 0])));
    size_of_sized = mem::size_of_val::<Ack<_>>(&x);
    let align_of_sized = mem::align_of_val::<Ack<_>>(&x);
    let y: Box<Ack<[u8]>> = x;
    size_of_unsized = mem::size_of_val::<Ack<_>>(&y);
    assert_eq!(size_of_sized, size_of_unsized);
    assert_eq!(align_of_sized, 1);
    assert_eq!(mem::align_of_val::<Ack<_>>(&y), 1);
}
