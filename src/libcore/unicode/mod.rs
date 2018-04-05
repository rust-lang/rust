// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![unstable(feature = "unicode", issue = "27783")]
#![allow(missing_docs)]

mod bool_trie;
pub(crate) mod tables;
pub(crate) mod version;

// For use in liballoc, not re-exported in libstd.
pub mod derived_property {
    pub use unicode::tables::derived_property::{Case_Ignorable, Cased};
}

// For use in libsyntax
pub mod property {
    pub use unicode::tables::property::Pattern_White_Space;
}

use iter::FusedIterator;

/// Iterator adaptor for encoding `char`s to UTF-16.
#[derive(Clone)]
#[allow(missing_debug_implementations)]
pub struct Utf16Encoder<I> {
    chars: I,
    extra: u16,
}

impl<I> Utf16Encoder<I> {
    /// Create a UTF-16 encoder from any `char` iterator.
    pub fn new(chars: I) -> Utf16Encoder<I>
        where I: Iterator<Item = char>
    {
        Utf16Encoder {
            chars,
            extra: 0,
        }
    }
}

impl<I> Iterator for Utf16Encoder<I>
    where I: Iterator<Item = char>
{
    type Item = u16;

    #[inline]
    fn next(&mut self) -> Option<u16> {
        if self.extra != 0 {
            let tmp = self.extra;
            self.extra = 0;
            return Some(tmp);
        }

        let mut buf = [0; 2];
        self.chars.next().map(|ch| {
            let n = ch.encode_utf16(&mut buf).len();
            if n == 2 {
                self.extra = buf[1];
            }
            buf[0]
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (low, high) = self.chars.size_hint();
        // every char gets either one u16 or two u16,
        // so this iterator is between 1 or 2 times as
        // long as the underlying iterator.
        (low, high.and_then(|n| n.checked_mul(2)))
    }
}

impl<I> FusedIterator for Utf16Encoder<I>
    where I: FusedIterator<Item = char> {}
