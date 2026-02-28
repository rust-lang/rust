//@ run-pass
#![feature(offset_of_enum)]
#![allow(unused)]

use std::mem::offset_of;

enum Never {}

#[repr(align(2))]
struct AlignedNever(Never);

enum Alpha {
    One(u8),
    Two(u8),
    Three(u8, u8, Never),
}

enum Beta {
    One(u8),
    Two(u8, Never),
}

enum Gamma {
    One(u32),
    Two(u8, u8, u8, Never),
}

enum Delta {
    One(u8, Never),
    Two(u8, u8, Never),
}

fn main() {
    assert!(offset_of!(Alpha, One.0) <= size_of::<Alpha>() - size_of::<u8>());
    assert!(offset_of!(Alpha, Two.0) <= size_of::<Alpha>() - size_of::<u8>());
    assert!(offset_of!(Alpha, Three.0) <= size_of::<Alpha>() - size_of::<u8>());
    assert!(offset_of!(Alpha, Three.1) <= size_of::<Alpha>() - size_of::<u8>());
    assert!(offset_of!(Alpha, Three.2) <= size_of::<Alpha>() - size_of::<Never>());
    assert!(offset_of!(Alpha, Three.0) != offset_of!(Alpha, Three.1));

    assert!(offset_of!(Beta, One.0) <= size_of::<Beta>() - size_of::<u8>());
    assert!(offset_of!(Beta, Two.0) <= size_of::<Beta>() - size_of::<u8>());
    assert!(offset_of!(Beta, Two.1) <= size_of::<Beta>() - size_of::<Never>());

    assert!(offset_of!(Gamma, One.0) <= size_of::<Gamma>() - size_of::<u32>());
    assert!(offset_of!(Gamma, Two.0) <= size_of::<Gamma>() - size_of::<u8>());
    assert!(offset_of!(Gamma, Two.1) <= size_of::<Gamma>() - size_of::<u8>());
    assert!(offset_of!(Gamma, Two.2) <= size_of::<Gamma>() - size_of::<u8>());
    assert!(offset_of!(Gamma, Two.3) <= size_of::<Gamma>() - size_of::<Never>());
    assert!(offset_of!(Gamma, Two.0) != offset_of!(Gamma, Two.1));
    assert!(offset_of!(Gamma, Two.0) != offset_of!(Gamma, Two.2));
    assert!(offset_of!(Gamma, Two.1) != offset_of!(Gamma, Two.2));

    assert!(offset_of!(Delta, One.0) <= size_of::<Delta>() - size_of::<u8>());
    assert!(offset_of!(Delta, One.1) <= size_of::<Delta>() - size_of::<Never>());
    assert!(offset_of!(Delta, Two.0) <= size_of::<Delta>() - size_of::<u8>());
    assert!(offset_of!(Delta, Two.0) <= size_of::<Delta>() - size_of::<u8>());
    assert!(offset_of!(Delta, Two.1) <= size_of::<Delta>() - size_of::<u8>());
    assert!(offset_of!(Delta, Two.2) <= size_of::<Delta>() - size_of::<Never>());
    assert!(offset_of!(Delta, Two.0) != offset_of!(Delta, Two.1));
}
