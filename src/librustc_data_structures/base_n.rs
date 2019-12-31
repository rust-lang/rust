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
    debug_assert!(base >= 2 && base <= MAX_BASE);
    let mut s = [0u8; 128];
    let mut index = 0;

    let base = base as u128;

    loop {
        s[index] = BASE_64[(n % base) as usize];
        index += 1;
        n /= base;

        if n == 0 {
            break;
        }
    }
    s[0..index].reverse();

    output.push_str(str::from_utf8(&s[0..index]).unwrap());
}

#[inline]
pub fn encode(n: u128, base: usize) -> String {
    let mut s = String::new();
    push_str(n, base, &mut s);
    s
}
