use std::sync::OnceLock;

use regex::bytes;

#[cfg(test)]
mod tests;

/// Given the raw contents of a string literal in LLVM IR assembly, decodes any
/// backslash escapes and returns a vector containing the resulting byte string.
pub(crate) fn unescape_llvm_string_contents(contents: &str) -> Vec<u8> {
    let escape_re = {
        static RE: OnceLock<bytes::Regex> = OnceLock::new();
        // LLVM IR supports two string escapes: `\\` and `\xx`.
        RE.get_or_init(|| bytes::Regex::new(r"\\\\|\\([0-9A-Za-z]{2})").unwrap())
    };

    fn u8_from_hex_digits(digits: &[u8]) -> u8 {
        // We know that the input contains exactly 2 hex digits, so these calls
        // should never fail.
        assert_eq!(digits.len(), 2);
        let digits = std::str::from_utf8(digits).unwrap();
        u8::from_str_radix(digits, 16).unwrap()
    }

    escape_re
        .replace_all(contents.as_bytes(), |captures: &bytes::Captures<'_>| {
            let byte = match captures.get(1) {
                None => b'\\',
                Some(hex_digits) => u8_from_hex_digits(hex_digits.as_bytes()),
            };
            [byte]
        })
        .into_owned()
}

/// LLVM's profiler/coverage metadata often uses an MD5 hash truncated to
/// 64 bits as a way to associate data stored in different tables/sections.
pub(crate) fn truncated_md5(bytes: &[u8]) -> u64 {
    use md5::{Digest, Md5};
    let mut hasher = Md5::new();
    hasher.update(bytes);
    let hash: [u8; 8] = hasher.finalize().as_slice()[..8].try_into().unwrap();
    // The truncated hash is explicitly little-endian, regardless of host
    // or target platform. (See `MD5Result::low` in LLVM's `MD5.h`.)
    u64::from_le_bytes(hash)
}
