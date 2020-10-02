#![allow(dead_code)]

#[cfg(test)]
mod tests;

use crate::fmt::{Formatter, Result, Write};
use core::str::lossy::{Utf8Lossy, Utf8LossyChunk};

pub fn debug_fmt_bytestring(slice: &[u8], f: &mut Formatter<'_>) -> Result {
    // Writes out a valid unicode string with the correct escape sequences
    fn write_str_escaped(f: &mut Formatter<'_>, s: &str) -> Result {
        for c in s.chars().flat_map(|c| c.escape_debug()) {
            f.write_char(c)?
        }
        Ok(())
    }

    f.write_str("\"")?;
    for Utf8LossyChunk { valid, broken } in Utf8Lossy::from_bytes(slice).chunks() {
        write_str_escaped(f, valid)?;
        for b in broken {
            write!(f, "\\x{:02X}", b)?;
        }
    }
    f.write_str("\"")
}
