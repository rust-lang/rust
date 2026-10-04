use std::borrow::Cow;
use std::sync::OnceLock;

use anyhow::{anyhow, ensure};
use regex::bytes;

use crate::parser::Parser;

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

impl<'a> Parser<'a> {
    /// Reads a sequence of:
    /// - Length of uncompressed data in bytes, as ULEB128
    /// - Length of compressed data in bytes (or 0), as ULEB128
    /// - The indicated number of compressed or uncompressed bytes
    ///
    /// If the number of compressed bytes is 0, the subsequent bytes are
    /// uncompressed. Otherwise, the subsequent bytes are compressed, and will
    /// be decompressed.
    ///
    /// Returns the uncompressed bytes that were read directly or decompressed.
    pub(crate) fn read_chunk_to_uncompressed_bytes(&mut self) -> anyhow::Result<Cow<'a, [u8]>> {
        let uncompressed_len = self.read_uleb128_usize()?;
        let compressed_len = self.read_uleb128_usize()?;

        if compressed_len == 0 {
            // The bytes are uncompressed, so read them directly.
            let uncompressed_bytes = self.read_n_bytes(uncompressed_len)?;
            Ok(Cow::Borrowed(uncompressed_bytes))
        } else {
            // The bytes are compressed, so read and decompress them.
            let compressed_bytes = self.read_n_bytes(compressed_len)?;

            let uncompressed_bytes = miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(
                compressed_bytes,
                uncompressed_len,
            )
            .map_err(|e| anyhow!("{e:?}"))?;
            ensure!(uncompressed_bytes.len() == uncompressed_len);

            Ok(Cow::Owned(uncompressed_bytes))
        }
    }
}
