// run-pass

// Check that unsizing does not change which field is considered for niche layout.

#![feature(offset_of)]
#![allow(dead_code)]

#[derive(Clone)]
struct WideptrField<T: ?Sized> {
    first: usize,
    second: usize,
    niche: NicheAtEnd,
    tail: T,
}

#[derive(Clone)]
#[repr(C)]
struct NicheAtEnd {
    arr: [u8; 7],
    b: bool,
}

type Tail = [bool; 8];

fn main() {
    assert_eq!(
        core::mem::offset_of!(WideptrField<Tail>, niche),
        core::mem::offset_of!(WideptrField<dyn Send>, niche)
    );
}
