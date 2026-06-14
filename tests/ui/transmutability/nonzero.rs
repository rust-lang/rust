#![feature(transmutability)]
#![allow(dead_code)]

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
}

fn main() {
    use std::num::{
        NonZero, NonZeroI8, NonZeroI16, NonZeroI32, NonZeroI64, NonZeroI128, NonZeroIsize,
        NonZeroU8, NonZeroU16, NonZeroU32, NonZeroU64, NonZeroU128, NonZeroUsize,
    };

    type NonZeroChar = NonZero<char>;

    assert::safety::<u8, NonZeroU8>(); //~ ERROR: cannot be safely transmuted
    assert::safety::<u16, NonZeroU16>(); //~ ERROR: cannot be safely transmuted
    assert::safety::<u32, NonZeroU32>(); //~ ERROR: cannot be safely transmuted
    assert::safety::<u64, NonZeroU64>(); //~ ERROR: cannot be safely transmuted
    assert::safety::<u128, NonZeroU128>(); //~ ERROR: cannot be safely transmuted
    assert::safety::<usize, NonZeroUsize>(); //~ ERROR: cannot be safely transmuted

    assert::safety::<i8, NonZeroI8>(); //~ ERROR: cannot be safely transmuted
    assert::safety::<i16, NonZeroI16>(); //~ ERROR: cannot be safely transmuted
    assert::safety::<i32, NonZeroI32>(); //~ ERROR: cannot be safely transmuted
    assert::safety::<i64, NonZeroI64>(); //~ ERROR: cannot be safely transmuted
    assert::safety::<i128, NonZeroI128>(); //~ ERROR: cannot be safely transmuted
    assert::safety::<isize, NonZeroIsize>(); //~ ERROR: cannot be safely transmuted

    assert::nothing::<NonZeroU8, NonZeroU8>(); //~ ERROR: cannot be safely transmuted
    assert::nothing::<NonZeroU16, NonZeroU16>(); //~ ERROR: cannot be safely transmuted
    assert::nothing::<NonZeroU32, NonZeroU32>(); //~ ERROR: cannot be safely transmuted
    assert::nothing::<NonZeroU64, NonZeroU64>(); //~ ERROR: cannot be safely transmuted
    assert::nothing::<NonZeroU128, NonZeroU128>(); //~ ERROR: cannot be safely transmuted
    assert::nothing::<NonZeroUsize, NonZeroUsize>(); //~ ERROR: cannot be safely transmuted

    assert::nothing::<NonZeroI8, NonZeroI8>(); //~ ERROR: cannot be safely transmuted
    assert::nothing::<NonZeroI16, NonZeroI16>(); //~ ERROR: cannot be safely transmuted
    assert::nothing::<NonZeroI32, NonZeroI32>(); //~ ERROR: cannot be safely transmuted
    assert::nothing::<NonZeroI64, NonZeroI64>(); //~ ERROR: cannot be safely transmuted
    assert::nothing::<NonZeroI128, NonZeroI128>(); //~ ERROR: cannot be safely transmuted
    assert::nothing::<NonZeroIsize, NonZeroIsize>(); //~ ERROR: cannot be safely transmuted

    assert::safety::<char, NonZeroChar>(); //~ ERROR: cannot be safely transmuted
    assert::safety::<u32, NonZeroChar>(); //~ ERROR: cannot be safely transmuted
    assert::safety::<NonZeroU32, NonZeroChar>(); //~ ERROR: cannot be safely transmuted
    assert::nothing::<NonZeroChar, NonZeroChar>(); //~ ERROR: cannot be safely transmuted

    assert::safety::<[u8; 3], [NonZeroU8; 3]>(); //~ ERROR: cannot be safely transmuted
    assert::safety::<[i16; 3], [NonZeroI16; 3]>(); //~ ERROR: cannot be safely transmuted
    assert::safety::<[char; 3], [NonZeroChar; 3]>(); //~ ERROR: cannot be safely transmuted

    assert::safety::<(u8, u16), (NonZeroU8, NonZeroU16)>(); //~ ERROR: cannot be safely transmuted
    assert::safety::<(char, u32), (NonZeroChar, NonZeroU32)>(); //~ ERROR: cannot be safely transmuted

    assert::safety::<u32, Option<NonZeroChar>>(); //~ ERROR: cannot be safely transmuted
    assert::safety::<NonZeroU32, Option<NonZeroChar>>(); //~ ERROR: cannot be safely transmuted

    assert::safety::<bool, NonZeroU8>(); //~ ERROR: cannot be safely transmuted
    assert::safety::<NonZeroU8, bool>(); //~ ERROR: cannot be safely transmuted
}
