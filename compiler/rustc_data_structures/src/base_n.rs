/// Converts unsigned integers into a string representation with some base.
/// Bases up to and including 36 can be used for case-insensitive things.
use std::str;

#[cfg(test)]
mod tests;

pub const MAX_BASE: usize = 64;
pub const ALPHANUMERIC_ONLY: usize = 62;
pub const CASE_INSENSITIVE: usize = 36;

const BASE_64: &[u8; MAX_BASE] =
    b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ@$";

#[inline]
pub fn push_str(mut n: u128, base: usize, output: &mut String) {
    debug_assert!(base >= 2 && base <= MAX_BASE);
    let mut s = [0u8; 128];
    let mut index = s.len();

    let base = base as u128;

    loop {
        index -= 1;
        s[index] = BASE_64[(n % base) as usize];
        n /= base;

        if n == 0 {
            break;
        }
    }

    output.push_str(unsafe {
        // SAFETY: `s` is populated using only valid utf8 characters from `BASE_64`
        str::from_utf8_unchecked(&s[index..])
    });
}

#[inline]
pub fn encode(n: u128, base: usize) -> String {
    let mut s = String::new();
    push_str(n, base, &mut s);
    s
}
