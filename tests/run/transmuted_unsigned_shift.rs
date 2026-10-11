// Compiler:
//
// Run-time:
//   status: 0

// cg_ssa gives a transmute to an unsigned integer the signed GCC type of the same size, so a
// backend that picks the kind of right shift from the GCC type sign-extends these values.

use std::hint::black_box;
use std::mem::transmute;
use std::num::NonZero;

macro_rules! check {
    ($ty:ty, $value:expr, $shift:expr, $expected:expr) => {{
        let value: $ty = unsafe { transmute(black_box(NonZero::<$ty>::new($value).unwrap())) };
        assert_eq!(value >> black_box($shift), $expected);
    }};
}

fn main() {
    check!(u8, 0x82, 4, 0x8);
    check!(u16, 0xa003, 12, 0xa);
    check!(u32, 0x8000_0001, 28, 0x8);
    check!(u64, 0x8000_0000_0000_0001, 60, 0x8);
    check!(u128, 0x8000_0000_0000_0000_0000_0000_0000_0001, 124, 0x8);
    check!(u128, 0x8000_0000_0000_0000_0000_0000_0000_0001, 60, 0x8_0000_0000_0000_0000);
}
