//! These functions compute the integer logarithm of their type, assuming
//! that someone has already checked that the value is strictly positive.

use crate::num::NonZero;

// 0 < val <= u8::MAX
#[inline]
const fn u8_impl(val: u8) -> u32 {
    let val = val as u32;

    // For better performance, avoid branches by assembling the solution
    // in the bits above the low 8 bits.

    // Adding c1 to val gives 10 in the top bits for val < 10, 11 for val >= 10
    const C1: u32 = 0b11_00000000 - 10; // 758
    // Adding c2 to val gives 01 in the top bits for val < 100, 10 for val >= 100
    const C2: u32 = 0b10_00000000 - 100; // 412

    // Value of top bits:
    //            +c1  +c2  1&2
    //     0..=9   10   01   00 = 0
    //   10..=99   11   01   01 = 1
    // 100..=255   11   10   10 = 2
    ((val + C1) & (val + C2)) >> 8
}

// 0 < val < 100_000
#[inline]
const fn less_than_5(val: u32) -> u32 {
    // Similar to u8, when adding one of these constants to val,
    // we get two possible bit patterns above the low 17 bits,
    // depending on whether val is below or above the threshold.
    const C1: u32 = 0b011_00000000000000000 - 10; // 393206
    const C2: u32 = 0b100_00000000000000000 - 100; // 524188
    const C3: u32 = 0b111_00000000000000000 - 1000; // 916504
    const C4: u32 = 0b100_00000000000000000 - 10000; // 514288

    // Value of top bits:
    //                +c1  +c2  1&2  +c3  +c4  3&4   ^
    //         0..=9  010  011  010  110  011  010  000 = 0
    //       10..=99  011  011  011  110  011  010  001 = 1
    //     100..=999  011  100  000  110  011  010  010 = 2
    //   1000..=9999  011  100  000  111  011  011  011 = 3
    // 10000..=99999  011  100  000  111  100  100  100 = 4
    (((val + C1) & (val + C2)) ^ ((val + C3) & (val + C4))) >> 17
}

// 0 < val <= u16::MAX
#[inline]
const fn u16_impl(val: u16) -> u32 {
    less_than_5(val as u32)
}

// 0 < val <= u32::MAX
#[inline]
const fn u32_impl(mut val: u32) -> u32 {
    let mut log = 0;
    if val >= 100_000 {
        val /= 100_000;
        log += 5;
    }
    log + less_than_5(val)
}

// 0 < val <= u64::MAX
#[inline]
const fn u64_impl(mut val: u64) -> u32 {
    let mut log = 0;
    if val >= 10_000_000_000 {
        val /= 10_000_000_000;
        log += 10;
    }
    if val >= 100_000 {
        val /= 100_000;
        log += 5;
    }
    log + less_than_5(val as u32)
}

// 0 < val <= u128::MAX
#[inline]
const fn u128_impl(mut val: u128) -> u32 {
    let mut log = 0;
    if val >= 100_000_000_000_000_000_000_000_000_000_000 {
        val /= 100_000_000_000_000_000_000_000_000_000_000;
        log += 32;
        return log + u32_impl(val as u32);
    }
    if val >= 10_000_000_000_000_000 {
        val /= 10_000_000_000_000_000;
        log += 16;
    }
    log + u64_impl(val as u64)
}

macro_rules! define_unsigned_ilog10 {
    ($($ty:ident => $impl_fn:ident,)*) => {$(
        #[inline]
        pub(super) const fn $ty(val: NonZero<$ty>) -> u32 {
            let result = $impl_fn(val.get());

            // SAFETY: Integer logarithm is monotonic non-decreasing, so the computed `result` cannot
            // exceed the value produced for the maximum input.
            unsafe { crate::hint::assert_unchecked(result <= const { $impl_fn($ty::MAX) }) };

            result
        }
    )*};
}

define_unsigned_ilog10! {
    u8 => u8_impl,
    u16 => u16_impl,
    u32 => u32_impl,
    u64 => u64_impl,
    u128 => u128_impl,
}

#[inline]
pub(super) const fn usize(val: NonZero<usize>) -> u32 {
    #[cfg(target_pointer_width = "16")]
    let impl_fn = u16;

    #[cfg(target_pointer_width = "32")]
    let impl_fn = u32;

    #[cfg(target_pointer_width = "64")]
    let impl_fn = u64;

    // SAFETY: We have selected the correct `impl_fn`, so the converting `val` to the argument is
    // safe.
    impl_fn(unsafe { NonZero::new_unchecked(val.get() as _) })
}

macro_rules! define_signed_ilog10 {
    ($($ty:ident => $impl_fn:ident,)*) => {$(
        // 0 < val <= $ty::MAX
        #[inline]
        pub(super) const fn $ty(val: $ty) -> Option<u32> {
            if val > 0 {
                let result = $impl_fn(val.cast_unsigned());

                // SAFETY: Integer logarithm is monotonic non-decreasing, so the computed `result`
                // cannot exceed the value produced for the maximum input.
                unsafe {
                    crate::hint::assert_unchecked(result <= const { $impl_fn($ty::MAX.cast_unsigned()) });
                }

                Some(result)
            } else {
                None
            }
        }
    )*};
}

define_signed_ilog10! {
    i8 => u8_impl,
    i16 => u16_impl,
    i32 => u32_impl,
    i64 => u64_impl,
    i128 => u128_impl,
}

/// Instantiate this panic logic once, rather than for all the ilog methods
/// on every single primitive type.
#[cold]
#[track_caller]
pub(super) const fn panic_for_nonpositive_argument() -> ! {
    panic!("argument of integer logarithm must be positive")
}
