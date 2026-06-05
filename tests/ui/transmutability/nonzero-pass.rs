//@ check-pass

#![feature(transmutability)]
#![feature(unsafe_fields)]
#![allow(dead_code, incomplete_features)]

mod assert {
    use std::mem::{Assume, TransmuteFrom};

    pub fn nothing<Src, Dst>()
    where
        Dst: TransmuteFrom<Src, { Assume::NOTHING }>,
    {
    }

    pub fn safety<Src, Dst>()
    where
        Dst: TransmuteFrom<Src, { Assume::SAFETY }>,
    {
    }

    pub fn safety_and_validity<Src, Dst>()
    where
        Dst: TransmuteFrom<Src, { Assume::SAFETY.and(Assume::VALIDITY) }>,
    {
    }
}

macro_rules! assert_integer_nonzero {
    ($int:ty, $nonzero:ty) => {
        assert::safety::<$nonzero, $int>();
        assert::safety::<$nonzero, $nonzero>();
        assert::safety_and_validity::<$int, $nonzero>();

        assert::safety::<Option<$nonzero>, $int>();
        assert::safety::<$int, Option<$nonzero>>();
        assert::safety::<Option<$nonzero>, Option<$nonzero>>();

        assert::safety::<[$nonzero; 3], [$int; 3]>();
        assert::safety_and_validity::<[$int; 3], [$nonzero; 3]>();
        assert::safety::<($nonzero, $nonzero), ($int, $int)>();
        assert::safety_and_validity::<($int, $int), ($nonzero, $nonzero)>();
    };
}

macro_rules! assert_same_width_integer_nonzeros {
    ($uint:ty, $int:ty, $nonzero_uint:ty, $nonzero_int:ty) => {
        assert::safety::<$nonzero_uint, $int>();
        assert::safety::<$nonzero_int, $uint>();
        assert::safety::<$nonzero_uint, $nonzero_int>();
        assert::safety::<$nonzero_int, $nonzero_uint>();

        assert::safety::<Option<$nonzero_uint>, $int>();
        assert::safety::<Option<$nonzero_int>, $uint>();
        assert::safety::<$uint, Option<$nonzero_int>>();
        assert::safety::<$int, Option<$nonzero_uint>>();
    };
}

#[repr(C)]
struct PublicField {
    pub field: u8,
}

#[repr(C)]
struct PublicUnsafeField {
    pub unsafe field: u8,
}

mod owner {
    #[repr(C)]
    pub struct VisibleFromChild {
        field: u8,
    }

    pub mod child {
        use super::VisibleFromChild;
        use crate::assert;

        pub fn check() {
            assert::nothing::<u8, VisibleFromChild>();
            assert::nothing::<VisibleFromChild, u8>();
        }
    }
}

fn main() {
    use std::num::{
        NonZero, NonZeroI8, NonZeroI16, NonZeroI32, NonZeroI64, NonZeroI128, NonZeroIsize,
        NonZeroU8, NonZeroU16, NonZeroU32, NonZeroU64, NonZeroU128, NonZeroUsize,
    };

    type NonZeroChar = NonZero<char>;

    assert_integer_nonzero!(u8, NonZeroU8);
    assert_integer_nonzero!(u16, NonZeroU16);
    assert_integer_nonzero!(u32, NonZeroU32);
    assert_integer_nonzero!(u64, NonZeroU64);
    assert_integer_nonzero!(u128, NonZeroU128);
    assert_integer_nonzero!(usize, NonZeroUsize);

    assert_integer_nonzero!(i8, NonZeroI8);
    assert_integer_nonzero!(i16, NonZeroI16);
    assert_integer_nonzero!(i32, NonZeroI32);
    assert_integer_nonzero!(i64, NonZeroI64);
    assert_integer_nonzero!(i128, NonZeroI128);
    assert_integer_nonzero!(isize, NonZeroIsize);

    assert_same_width_integer_nonzeros!(u8, i8, NonZeroU8, NonZeroI8);
    assert_same_width_integer_nonzeros!(u16, i16, NonZeroU16, NonZeroI16);
    assert_same_width_integer_nonzeros!(u32, i32, NonZeroU32, NonZeroI32);
    assert_same_width_integer_nonzeros!(u64, i64, NonZeroU64, NonZeroI64);
    assert_same_width_integer_nonzeros!(u128, i128, NonZeroU128, NonZeroI128);
    assert_same_width_integer_nonzeros!(usize, isize, NonZeroUsize, NonZeroIsize);

    assert::safety::<NonZeroChar, char>();
    assert::safety::<NonZeroChar, u32>();
    assert::safety::<NonZeroChar, NonZeroU32>();
    assert::safety::<Option<NonZeroChar>, char>();
    assert::safety::<char, Option<NonZeroChar>>();
    assert::safety::<Option<NonZeroChar>, u32>();
    assert::safety::<Option<NonZeroChar>, Option<NonZeroChar>>();
    assert::safety_and_validity::<char, NonZeroChar>();
    assert::safety_and_validity::<u32, NonZeroChar>();

    assert::safety::<[NonZeroChar; 3], [char; 3]>();
    assert::safety::<[NonZeroChar; 3], [u32; 3]>();
    assert::safety_and_validity::<[char; 3], [NonZeroChar; 3]>();
    assert::safety::<(NonZeroChar, NonZeroChar), (char, char)>();

    assert::nothing::<u8, PublicField>();
    assert::nothing::<PublicField, u8>();
    owner::child::check();

    assert::safety::<u8, PublicUnsafeField>();
    assert::safety::<PublicUnsafeField, u8>();
}
