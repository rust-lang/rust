/// Converts unsigned integers into a string representation with some base.
/// Bases up to and including 36 can be used for case-insensitive things.
use std::str;

#[cfg(test)]
mod tests;

pub const MAX_BASE: usize = 64;
pub const ALPHANUMERIC_ONLY: usize = 62;
pub const CASE_INSENSITIVE: usize = 36;

const BASE_64: &[u8; MAX_BASE as usize] =
    b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ@$";

#[inline]
pub fn push_str(mut n: u128, base: usize, output: &mut String) {
    assert!(base >= 2 && base <= MAX_BASE);
    let mut s = [0u8; 128];
    let mut first_index = 0;

    let base = base as u128;

    for idx in (0..128).rev() {
        // SAFETY: given `base <= MAX_BASE`, so `n % base < MAX_BASE`
        s[idx] = unsafe { *BASE_64.get_unchecked((n % base) as usize) };
        n /= base;

        if n == 0 {
            first_index = idx;
            break;
        }
    }

    // SAFETY: all chars in given range is nonnull ascii
    output.push_str(unsafe { str::from_utf8_unchecked(&s[first_index..]) });
}

#[inline]
pub fn encode(n: u128, base: usize) -> String {
    let mut s = String::new();
    push_str(n, base, &mut s);
    s
}
