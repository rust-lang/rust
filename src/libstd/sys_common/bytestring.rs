// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]

use fmt::{Formatter, Result, Write};
use std_unicode::lossy::{Utf8Lossy, Utf8LossyChunk};

pub fn debug_fmt_bytestring(slice: &[u8], f: &mut Formatter) -> Result {
    // Writes out a valid unicode string with the correct escape sequences
    fn write_str_escaped(f: &mut Formatter, s: &str) -> Result {
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

#[cfg(test)]
mod tests {
    use super::*;
    use fmt::{Formatter, Result, Debug};

    #[test]
    fn smoke() {
        struct Helper<'a>(&'a [u8]);

        impl<'a> Debug for Helper<'a> {
            fn fmt(&self, f: &mut Formatter) -> Result {
                debug_fmt_bytestring(self.0, f)
            }
        }

        let input =      b"\xF0hello,\tworld";
        let expected = r#""\xF0hello,\tworld""#;
        let output = format!("{:?}", Helper(input));

        assert!(output == expected);
    }
}
