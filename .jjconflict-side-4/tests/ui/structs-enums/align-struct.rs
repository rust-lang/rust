//@ run-pass
#![allow(dead_code, unused_allocation)]

use std::mem;

// Raising alignment
#[repr(align(16))]
#[derive(Clone, Copy, Debug)]
struct Align16(i32);

// Lowering has no effect
#[repr(align(1))]
struct Align1(i32);

// Multiple attributes take the max
#[repr(align(4))]
#[repr(align(16))]
#[repr(align(8))]
struct AlignMany(i32);

// Raising alignment may not alter size.
#[repr(align(8))]
struct Align8Many {
    a: i32,
    b: i32,
    c: i32,
    d: u8,
}

enum Enum {
    A(i32),
    B(Align16),
}

// Nested alignment - use `#[repr(C)]` to suppress field reordering for sizeof test
#[repr(C)]
struct Nested {
    a: i32,
    b: i32,
    c: Align16,
    d: i8,
}

#[repr(packed)]
struct Packed(i32);

#[repr(align(16))]
struct AlignContainsPacked {
    a: Packed,
    b: Packed,
}

#[repr(C, packed(4))]
struct Packed4C {
    a: u32,
    b: u64,
}

#[repr(align(16))]
struct AlignContainsPacked4C {
    a: Packed4C,
    b: u64,
}

// The align limit was originally smaller (2^15).
// Check that it works with big numbers.
#[repr(align(0x10000))]
struct AlignLarge {
    stuff: [u8; 0x10000],
}

union UnionContainsAlign {
    a: Align16,
    b: f32,
}

impl Align16 {
    // return aligned type
    pub fn new(i: i32) -> Align16 {
        Align16(i)
    }
    // pass aligned type
    pub fn consume(a: Align16) -> i32 {
        a.0
    }
}

const CONST_ALIGN16: Align16 = Align16(7);
static STATIC_ALIGN16: Align16 = Align16(8);

// Check the actual address is aligned
fn is_aligned_to<T>(p: &T, align: usize) -> bool {
    let addr = p as *const T as usize;
    (addr & (align - 1)) == 0
}

pub fn main() {
    // check alignment and size by type and value
    assert_eq!(mem::align_of::<Align16>(), 16);
    assert_eq!(mem::size_of::<Align16>(), 16);

    let a = Align16(7);
    assert_eq!(a.0, 7);
    assert_eq!(mem::align_of_val(&a), 16);
    assert_eq!(mem::size_of_val(&a), 16);

    assert!(is_aligned_to(&a, 16));

    // lowering should have no effect
    assert_eq!(mem::align_of::<Align1>(), 4);
    assert_eq!(mem::size_of::<Align1>(), 4);
    let a = Align1(7);
    assert_eq!(a.0, 7);
    assert_eq!(mem::align_of_val(&a), 4);
    assert_eq!(mem::size_of_val(&a), 4);
    assert!(is_aligned_to(&a, 4));

    // when multiple attributes are specified the max should be used
    assert_eq!(mem::align_of::<AlignMany>(), 16);
    assert_eq!(mem::size_of::<AlignMany>(), 16);
    let a = AlignMany(7);
    assert_eq!(a.0, 7);
    assert_eq!(mem::align_of_val(&a), 16);
    assert_eq!(mem::size_of_val(&a), 16);
    assert!(is_aligned_to(&a, 16));

    // raising alignment should not reduce size
    assert_eq!(mem::align_of::<Align8Many>(), 8);
    assert_eq!(mem::size_of::<Align8Many>(), 16);
    let a = Align8Many { a: 1, b: 2, c: 3, d: 4 };
    assert_eq!(a.a, 1);
    assert_eq!(mem::align_of_val(&a), 8);
    assert_eq!(mem::size_of_val(&a), 16);
    assert!(is_aligned_to(&a, 8));

    // return type
    let a = Align16::new(1);
    assert_eq!(mem::align_of_val(&a), 16);
    assert_eq!(mem::size_of_val(&a), 16);
    assert_eq!(a.0, 1);
    assert!(is_aligned_to(&a, 16));
    assert_eq!(Align16::consume(a), 1);

    // check const alignment, size and value
    assert_eq!(mem::align_of_val(&CONST_ALIGN16), 16);
    assert_eq!(mem::size_of_val(&CONST_ALIGN16), 16);
    assert_eq!(CONST_ALIGN16.0, 7);
    assert!(is_aligned_to(&CONST_ALIGN16, 16));

    // check global static alignment, size and value
    assert_eq!(mem::align_of_val(&STATIC_ALIGN16), 16);
    assert_eq!(mem::size_of_val(&STATIC_ALIGN16), 16);
    assert_eq!(STATIC_ALIGN16.0, 8);
    assert!(is_aligned_to(&STATIC_ALIGN16, 16));

    // Note that the size of Nested may change if struct field re-ordering is enabled
    assert_eq!(mem::align_of::<Nested>(), 16);
    assert_eq!(mem::size_of::<Nested>(), 48);
    let a = Nested { a: 1, b: 2, c: Align16(3), d: 4 };
    assert_eq!(mem::align_of_val(&a), 16);
    assert_eq!(mem::align_of_val(&a.b), 4);
    assert_eq!(mem::align_of_val(&a.c), 16);
    assert_eq!(mem::size_of_val(&a), 48);
    assert!(is_aligned_to(&a, 16));
    // check the correct fields are indexed
    assert_eq!(a.a, 1);
    assert_eq!(a.b, 2);
    assert_eq!(a.c.0, 3);
    assert_eq!(a.d, 4);

    // enum should be aligned to max alignment
    assert_eq!(mem::align_of::<Enum>(), 16);
    assert_eq!(mem::align_of_val(&Enum::B(Align16(0))), 16);
    let e = Enum::B(Align16(15));
    match e {
        Enum::B(ref a) => {
            assert_eq!(a.0, 15);
            assert_eq!(mem::align_of_val(a), 16);
            assert_eq!(mem::size_of_val(a), 16);
        }
        _ => (),
    }
    assert!(is_aligned_to(&e, 16));

    // check union alignment
    assert_eq!(mem::align_of::<UnionContainsAlign>(), 16);
    assert_eq!(mem::size_of::<UnionContainsAlign>(), 16);
    let u = UnionContainsAlign { a: Align16(10) };
    unsafe {
        assert_eq!(mem::align_of_val(&u.a), 16);
        assert_eq!(mem::size_of_val(&u.a), 16);
        assert_eq!(u.a.0, 10);
        let UnionContainsAlign { a } = u;
        assert_eq!(a.0, 10);
    }

    // arrays of aligned elements should also be aligned
    assert_eq!(mem::align_of::<[Align16; 2]>(), 16);
    assert_eq!(mem::size_of::<[Align16; 2]>(), 32);

    let a = [Align16(0), Align16(1)];
    assert_eq!(mem::align_of_val(&a[0]), 16);
    assert_eq!(mem::align_of_val(&a[1]), 16);
    assert!(is_aligned_to(&a, 16));

    // check heap value is aligned
    assert_eq!(mem::align_of_val(Box::new(Align16(0)).as_ref()), 16);

    // check heap array is aligned
    let a = vec![Align16(0), Align16(1)];
    assert_eq!(mem::align_of_val(&a[0]), 16);
    assert_eq!(mem::align_of_val(&a[1]), 16);

    assert_eq!(mem::align_of::<AlignContainsPacked>(), 16);
    assert_eq!(mem::size_of::<AlignContainsPacked>(), 16);
    let a = AlignContainsPacked { a: Packed(1), b: Packed(2) };
    assert_eq!(mem::align_of_val(&a), 16);
    assert_eq!(mem::align_of_val(&a.a), 1);
    assert_eq!(mem::align_of_val(&a.b), 1);
    assert_eq!(mem::size_of_val(&a), 16);
    assert!(is_aligned_to(&a, 16));

    assert_eq!(mem::align_of::<AlignContainsPacked4C>(), 16);
    assert_eq!(mem::size_of::<AlignContainsPacked4C>(), 32);
    let a = AlignContainsPacked4C { a: Packed4C { a: 1, b: 2 }, b: 3 };
    assert_eq!(mem::align_of_val(&a), 16);
    assert_eq!(mem::align_of_val(&a.a), 4);
    assert_eq!(mem::align_of_val(&a.b), mem::align_of::<u64>());
    assert_eq!(mem::size_of_val(&a), 32);
    assert!(is_aligned_to(&a, 16));

    let mut large = Box::new(AlignLarge { stuff: [0; 0x10000] });
    large.stuff[0] = 132;
    *large.stuff.last_mut().unwrap() = 102;
    assert_eq!(large.stuff[0], 132);
    assert_eq!(large.stuff.last(), Some(&102));
    assert_eq!(mem::align_of::<AlignLarge>(), 0x10000);
    assert_eq!(mem::align_of_val(&*large), 0x10000);
    assert!(is_aligned_to(&*large, 0x10000));
}
