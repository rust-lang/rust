#[cfg(test)]
mod tests;

use anyhow::ensure;
use regex::bytes;
use std::sync::OnceLock;

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

pub(crate) struct Parser<'a> {
    rest: &'a [u8],
}

impl<'a> Parser<'a> {
    pub(crate) fn new(input: &'a [u8]) -> Self {
        Self { rest: input }
    }

    pub(crate) fn ensure_empty(self) -> anyhow::Result<()> {
        ensure!(self.rest.is_empty(), "unparsed bytes: 0x{:02x?}", self.rest);
        Ok(())
    }

    pub(crate) fn read_n_bytes(&mut self, n: usize) -> anyhow::Result<&'a [u8]> {
        ensure!(n <= self.rest.len());

        let (bytes, rest) = self.rest.split_at(n);
        self.rest = rest;
        Ok(bytes)
    }

    pub(crate) fn read_uleb128_u32(&mut self) -> anyhow::Result<u32> {
        self.read_uleb128_u64_and_convert()
    }

    pub(crate) fn read_uleb128_usize(&mut self) -> anyhow::Result<usize> {
        self.read_uleb128_u64_and_convert()
    }

    fn read_uleb128_u64_and_convert<T>(&mut self) -> anyhow::Result<T>
    where
        T: TryFrom<u64> + 'static,
        T::Error: std::error::Error + Send + Sync,
    {
        let mut temp_rest = self.rest;
        let raw_value: u64 = leb128::read::unsigned(&mut temp_rest)?;
        let converted_value = T::try_from(raw_value)?;

        // Only update `self.rest` if the above steps succeeded, so that the
        // parser position can be used for error reporting if desired.
        self.rest = temp_rest;
        Ok(converted_value)
    }
}
