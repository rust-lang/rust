// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;

// Provide some more formatting options for some data types (at the moment
// that's just `{:x}` for slices of u8).

pub struct FmtWrap<T>(pub T);

impl<'a> fmt::LowerHex for FmtWrap<&'a [u8]> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        for byte in self.0.iter() {
            try!(write!(formatter, "{:02x}", byte));
        }
        Ok(())
    }
}

#[test]
fn test_lower_hex() {
    let bytes: &[u8] = &[0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef];
    assert_eq!("0123456789abcdef", &format!("{:x}", FmtWrap(bytes)));
}
