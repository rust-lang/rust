//@ run-pass
#![allow(dead_code)]

use std::mem;

// Get around the limitations of CTFE in today's Rust.
const fn choice_u64(c: bool, a: u64, b: u64) -> u64 {
    (-(c as i64) as u64) & a | (-(!c as i64) as u64) & b
}

const fn max_usize(a: usize, b: usize) -> usize {
    choice_u64(a > b, a as u64, b as u64) as usize
}

const fn align_to(size: usize, align: usize) -> usize {
    (size + (align - 1)) & !(align - 1)
}

const fn packed_union_size_of<A, B>() -> usize {
    max_usize(mem::size_of::<A>(), mem::size_of::<B>())
}

const fn union_align_of<A, B>() -> usize {
    max_usize(mem::align_of::<A>(), mem::align_of::<B>())
}

const fn union_size_of<A, B>() -> usize {
    align_to(packed_union_size_of::<A, B>(), union_align_of::<A, B>())
}

macro_rules! fake_union {
    ($name:ident { $a:ty, $b:ty }) => (
        struct $name {
            _align: ([$a; 0], [$b; 0]),
            _bytes: [u8; union_size_of::<$a, $b>()]
        }
    )
}

// Check that we can (poorly) emulate unions by
// calling size_of and align_of at compile-time.
fake_union!(U { u16, [u8; 3] });

fn test(u: U) {
    assert_eq!(mem::size_of_val(&u._bytes), 4);
}

fn main() {
    assert_eq!(mem::size_of::<U>(), 4);
    assert_eq!(mem::align_of::<U>(), 2);
}
