//! Bit manipulation utilities

/// Sets the `bit` of `x`.
pub const fn set(x: usize, bit: u32) -> usize {
    x | 1 << bit
}

/// Tests the `bit` of `x`.
pub const fn test(x: usize, bit: u32) -> bool {
    x & (1 << bit) != 0
}
