//@ revisions: old next
//@ [next] compile-flags: -Znext-solver
//@ run-pass
#![feature(field_projections, freeze, unsafe_unpin)]
#![expect(incomplete_features, dead_code)]
use std::field::field_of;
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::{Freeze, Unpin, UnsafeUnpin};
use std::panic::{RefUnwindSafe, UnwindSafe};

struct Struct {
    field: u32,
    tail: [u32],
}

union Union {
    field: u32,
}

enum Enum {
    Variant1 { field: u32 },
    Variant2(u32),
}

type Tuple = ((), usize, String, dyn Debug);

fn assert_traits<
    T: Sized
        + Freeze
        + RefUnwindSafe
        + Send
        + Sync
        + Unpin
        + UnsafeUnpin
        + UnwindSafe
        + Copy
        + Debug
        + Default
        + Eq
        + Hash
        + Ord,
>() {
}

fn main() {
    assert_traits::<field_of!(Struct, field)>();
    assert_traits::<field_of!(Struct, tail)>();

    assert_traits::<field_of!(Union, field)>();

    assert_traits::<field_of!(Enum, Variant1.field)>();
    assert_traits::<field_of!(Enum, Variant2.0)>();

    assert_traits::<field_of!(Tuple, 0)>();
    assert_traits::<field_of!(Tuple, 1)>();
    assert_traits::<field_of!(Tuple, 2)>();
    assert_traits::<field_of!(Tuple, 3)>();
}
