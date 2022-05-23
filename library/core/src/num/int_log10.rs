/// These functions compute the integer logarithm of their type, assuming
/// that someone has already checked that the the value is strictly positive.

// 0 < val <= u8::MAX
#[inline]
pub const fn u8(val: u8) -> u32 {
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
pub const fn u16(val: u16) -> u32 {
    less_than_5(val as u32)
}

// 0 < val <= u32::MAX
#[inline]
pub const fn u32(mut val: u32) -> u32 {
    let mut log = 0;
    if val >= 100_000 {
        val /= 100_000;
        log += 5;
    }
    log + less_than_5(val)
}

// 0 < val <= u64::MAX
#[inline]
pub const fn u64(mut val: u64) -> u32 {
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
pub const fn u128(mut val: u128) -> u32 {
    let mut log = 0;
    if val >= 100_000_000_000_000_000_000_000_000_000_000 {
        val /= 100_000_000_000_000_000_000_000_000_000_000;
        log += 32;
        return log + u32(val as u32);
    }
    if val >= 10_000_000_000_000_000 {
        val /= 10_000_000_000_000_000;
        log += 16;
    }
    log + u64(val as u64)
}

#[cfg(target_pointer_width = "16")]
#[inline]
pub const fn usize(val: usize) -> u32 {
    u16(val as _)
}

#[cfg(target_pointer_width = "32")]
#[inline]
pub const fn usize(val: usize) -> u32 {
    u32(val as _)
}

#[cfg(target_pointer_width = "64")]
#[inline]
pub const fn usize(val: usize) -> u32 {
    u64(val as _)
}

// 0 < val <= i8::MAX
#[inline]
pub const fn i8(val: i8) -> u32 {
    u8(val as u8)
}

// 0 < val <= i16::MAX
#[inline]
pub const fn i16(val: i16) -> u32 {
    u16(val as u16)
}

// 0 < val <= i32::MAX
#[inline]
pub const fn i32(val: i32) -> u32 {
    u32(val as u32)
}

// 0 < val <= i64::MAX
#[inline]
pub const fn i64(val: i64) -> u32 {
    u64(val as u64)
}

// 0 < val <= i128::MAX
#[inline]
pub const fn i128(val: i128) -> u32 {
    u128(val as u128)
}
