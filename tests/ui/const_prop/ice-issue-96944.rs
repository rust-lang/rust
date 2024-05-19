//@ build-pass
#![crate_type = "lib"]
#![allow(arithmetic_overflow)]

pub trait BitSplit {
    type Half;
    fn merge(halves: [Self::Half; 2]) -> Self;
}

macro_rules! impl_ints {
    ($int:ty => $half:ty; $mask:expr) => {
        impl BitSplit for $int {
            type Half = $half;
            #[inline]
            fn merge(halves: [Self::Half; 2]) -> Self {
                const HALF_SIZE: usize = std::mem::size_of::<$half>() * 8;
                (halves[0] << HALF_SIZE) as $int | halves[1] as $int
            }
        }
    };
}

impl_ints!(u128 => u64; 0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF);
impl_ints!( u64 => u32;                     0x0000_0000_FFFF_FFFF);
impl_ints!( u32 => u16;                               0x0000_FFFF);
impl_ints!( u16 =>  u8;                                    0x00FF);
