//@ run-pass
#![allow(dead_code)]

use std::mem;

// Raising alignment
#[repr(align(16))]
enum Align16 {
    Foo { foo: u32 },
    Bar { bar: u32 },
}

// Raise alignment by maximum
#[repr(align(1), align(16))]
#[repr(align(32))]
#[repr(align(4))]
enum Align32 {
    Foo,
    Bar,
}

// Not reducing alignment
#[repr(align(4))]
enum AlsoAlign16 {
    Foo { limb_with_align16: Align16 },
    Bar,
}

// No niche for discriminant when used as limb
#[repr(align(16))]
struct NoNiche16(u64, u64);

// Discriminant will require extra space, but enum needs to stay compatible
// with alignment 16
#[repr(align(1))]
enum AnotherAlign16 {
    Foo { limb_with_noniche16: NoNiche16 },
    Bar,
    Baz,
}

fn main() {
    assert_eq!(mem::align_of::<Align16>(), 16);
    assert_eq!(mem::size_of::<Align16>(), 16);

    assert_eq!(mem::align_of::<Align32>(), 32);
    assert_eq!(mem::size_of::<Align32>(), 32);

    assert_eq!(mem::align_of::<AlsoAlign16>(), 16);
    assert_eq!(mem::size_of::<AlsoAlign16>(), 16);

    assert_eq!(mem::align_of::<AnotherAlign16>(), 16);
    assert_eq!(mem::size_of::<AnotherAlign16>(), 32);
}
