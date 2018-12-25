// run-pass
#![allow(dead_code)]
use std::mem::size_of;

enum E {
    A = 1,
    B = 2,
    C = 3,
}

struct S {
    a: u16,
    b: u8,
    e: E,
}

fn main() {
    assert_eq!(size_of::<E>(), 1);
    assert_eq!(size_of::<Option<E>>(), 1);
    assert_eq!(size_of::<Result<E, ()>>(), 1);
    assert_eq!(size_of::<Option<S>>(), size_of::<S>());
    let enone = None::<E>;
    let esome = Some(E::A);
    if let Some(..) = enone {
        panic!();
    }
    if let None = esome {
        panic!();
    }
}
