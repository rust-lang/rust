// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Really Bad Markup Language (rbml) is an internal serialization format of rustc.
//! This is not intended to be used by users.
//!
//! Originally based on the Extensible Binary Markup Language
//! (ebml; http://www.matroska.org/technical/specs/rfc/index.html),
//! it is now a separate format tuned for the rust object metadata.
//!
//! # Encoding
//!
//! RBML document consists of the tag, length and data.
//! The encoded data can contain multiple RBML documents concatenated.
//!
//! **Tags** are a hint for the following data.
//! Tags are a number from 0x000 to 0xfff, where 0xf0 through 0xff is reserved.
//! Tags less than 0xf0 are encoded in one literal byte.
//! Tags greater than 0xff are encoded in two big-endian bytes,
//! where the tag number is ORed with 0xf000. (E.g. tag 0x123 = `f1 23`)
//!
//! **Lengths** encode the length of the following data.
//! It is a variable-length unsigned isize, and one of the following forms:
//!
//! - `80` through `fe` for lengths up to 0x7e;
//! - `40 ff` through `7f ff` for lengths up to 0x3fff;
//! - `20 40 00` through `3f ff ff` for lengths up to 0x1fffff;
//! - `10 20 00 00` through `1f ff ff ff` for lengths up to 0xfffffff.
//!
//! The "overlong" form is allowed so that the length can be encoded
//! without the prior knowledge of the encoded data.
//! For example, the length 0 can be represented either by `80`, `40 00`,
//! `20 00 00` or `10 00 00 00`.
//! The encoder tries to minimize the length if possible.
//! Also, some predefined tags listed below are so commonly used that
//! their lengths are omitted ("implicit length").
//!
//! **Data** can be either binary bytes or zero or more nested RBML documents.
//! Nested documents cannot overflow, and should be entirely contained
//! within a parent document.

#[cfg(test)]
use test::Bencher;

use std::fmt;
use std::str;

macro_rules! try_or {
    ($e:expr, $r:expr) => (
        match $e {
            Ok(x) => x,
            Err(_) => return $r
        }
    )
}

#[derive(Clone, Copy)]
pub struct Doc<'a> {
    pub data: &'a [u8],
    pub start: usize,
    pub end: usize,
}

impl<'doc> Doc<'doc> {
    pub fn new(data: &'doc [u8]) -> Doc<'doc> {
        Doc {
            data: data,
            start: 0,
            end: data.len(),
        }
    }

    pub fn at(data: &'doc [u8], start: usize) -> Doc<'doc> {
        let elt_tag = tag_at(data, start).unwrap();
        let elt_size = tag_len_at(data, elt_tag.next).unwrap();
        let end = elt_size.next + elt_size.val;
        Doc {
            data: data,
            start: elt_size.next,
            end: end,
        }
    }

    pub fn maybe_child(&self, tag: usize) -> Option<Doc<'doc>> {
        let mut pos = self.start;
        while pos < self.end {
            let elt_tag = try_or!(tag_at(self.data, pos), None);
            let elt_size = try_or!(tag_len_at(self.data, elt_tag.next), None);
            pos = elt_size.next + elt_size.val;
            if elt_tag.val == tag {
                return Some(Doc {
                    data: self.data,
                    start: elt_size.next,
                    end: pos,
                });
            }
        }
        None
    }

    pub fn child(&self, tag: usize) -> Doc<'doc> {
        match self.maybe_child(tag) {
            Some(d) => d,
            None => {
                bug!("failed to find child with tag {:?}", tag);
            }
        }
    }

    pub fn children_of(&self, tag: usize) -> DocsIterator<'doc> {
        DocsIterator { d: self.child(tag) }
    }
}

#[derive(Debug)]
pub enum Error {
    IntTooBig(usize),
    InvalidTag(usize)
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // FIXME: this should be a more useful display form
        fmt::Debug::fmt(self, f)
    }
}

#[derive(Copy, Clone)]
struct Res {
    val: usize,
    next: usize,
}

fn tag_at(data: &[u8], start: usize) -> Result<Res, Error> {
    let v = data[start] as usize;
    if v < 0xf0 {
        Ok(Res {
            val: v,
            next: start + 1,
        })
    } else if v > 0xf0 {
        Ok(Res {
            val: ((v & 0xf) << 8) | data[start + 1] as usize,
            next: start + 2,
        })
    } else {
        // every tag starting with byte 0xf0 is an overlong form, which is prohibited.
        Err(Error::InvalidTag(v))
    }
}

#[inline(never)]
fn vuint_at_slow(data: &[u8], start: usize) -> Result<Res, Error> {
    let a = data[start];
    if a & 0x80 != 0 {
        return Ok(Res {
            val: (a & 0x7f) as usize,
            next: start + 1,
        });
    }
    if a & 0x40 != 0 {
        return Ok(Res {
            val: ((a & 0x3f) as usize) << 8 | (data[start + 1] as usize),
            next: start + 2,
        });
    }
    if a & 0x20 != 0 {
        return Ok(Res {
            val: ((a & 0x1f) as usize) << 16 | (data[start + 1] as usize) << 8 |
                 (data[start + 2] as usize),
            next: start + 3,
        });
    }
    if a & 0x10 != 0 {
        return Ok(Res {
            val: ((a & 0x0f) as usize) << 24 | (data[start + 1] as usize) << 16 |
                 (data[start + 2] as usize) << 8 |
                 (data[start + 3] as usize),
            next: start + 4,
        });
    }
    Err(Error::IntTooBig(a as usize))
}

fn vuint_at(data: &[u8], start: usize) -> Result<Res, Error> {
    if data.len() - start < 4 {
        return vuint_at_slow(data, start);
    }

    // Lookup table for parsing EBML Element IDs as per
    // http://ebml.sourceforge.net/specs/ The Element IDs are parsed by
    // reading a big endian u32 positioned at data[start].  Using the four
    // most significant bits of the u32 we lookup in the table below how
    // the element ID should be derived from it.
    //
    // The table stores tuples (shift, mask) where shift is the number the
    // u32 should be right shifted with and mask is the value the right
    // shifted value should be masked with.  If for example the most
    // significant bit is set this means it's a class A ID and the u32
    // should be right shifted with 24 and masked with 0x7f. Therefore we
    // store (24, 0x7f) at index 0x8 - 0xF (four bit numbers where the most
    // significant bit is set).
    //
    // By storing the number of shifts and masks in a table instead of
    // checking in order if the most significant bit is set, the second
    // most significant bit is set etc. we can replace up to three
    // "and+branch" with a single table lookup which gives us a measured
    // speedup of around 2x on x86_64.
    static SHIFT_MASK_TABLE: [(usize, u32); 16] = [(0, 0x0),
                                                   (0, 0x0fffffff),
                                                   (8, 0x1fffff),
                                                   (8, 0x1fffff),
                                                   (16, 0x3fff),
                                                   (16, 0x3fff),
                                                   (16, 0x3fff),
                                                   (16, 0x3fff),
                                                   (24, 0x7f),
                                                   (24, 0x7f),
                                                   (24, 0x7f),
                                                   (24, 0x7f),
                                                   (24, 0x7f),
                                                   (24, 0x7f),
                                                   (24, 0x7f),
                                                   (24, 0x7f)];

    unsafe {
        let ptr = data.as_ptr().offset(start as isize) as *const u32;
        let val = u32::from_be(*ptr);

        let i = (val >> 28) as usize;
        let (shift, mask) = SHIFT_MASK_TABLE[i];
        Ok(Res {
            val: ((val >> shift) & mask) as usize,
            next: start + ((32 - shift) >> 3),
        })
    }
}

fn tag_len_at(data: &[u8], next: usize) -> Result<Res, Error> {
    vuint_at(data, next)
}

pub struct DocsIterator<'a> {
    d: Doc<'a>,
}

impl<'a> Iterator for DocsIterator<'a> {
    type Item = Doc<'a>;

    fn next(&mut self) -> Option<Doc<'a>> {
        if self.d.start >= self.d.end {
            return None;
        }

        let elt_tag = try_or!(tag_at(self.d.data, self.d.start), {
            self.d.start = self.d.end;
            None
        });
        let elt_size = try_or!(tag_len_at(self.d.data, elt_tag.next), {
            self.d.start = self.d.end;
            None
        });

        let end = elt_size.next + elt_size.val;
        let doc = Doc {
            data: self.d.data,
            start: elt_size.next,
            end: end,
        };

        self.d.start = end;
        return Some(doc);
    }
}

#[test]
fn test_vuint_at() {
    let data = &[
        0x80,
        0xff,
        0x40, 0x00,
        0x7f, 0xff,
        0x20, 0x00, 0x00,
        0x3f, 0xff, 0xff,
        0x10, 0x00, 0x00, 0x00,
        0x1f, 0xff, 0xff, 0xff
    ];

    let mut res: Res;

    // Class A
    res = vuint_at(data, 0).unwrap();
    assert_eq!(res.val, 0);
    assert_eq!(res.next, 1);
    res = vuint_at(data, res.next).unwrap();
    assert_eq!(res.val, (1 << 7) - 1);
    assert_eq!(res.next, 2);

    // Class B
    res = vuint_at(data, res.next).unwrap();
    assert_eq!(res.val, 0);
    assert_eq!(res.next, 4);
    res = vuint_at(data, res.next).unwrap();
    assert_eq!(res.val, (1 << 14) - 1);
    assert_eq!(res.next, 6);

    // Class C
    res = vuint_at(data, res.next).unwrap();
    assert_eq!(res.val, 0);
    assert_eq!(res.next, 9);
    res = vuint_at(data, res.next).unwrap();
    assert_eq!(res.val, (1 << 21) - 1);
    assert_eq!(res.next, 12);

    // Class D
    res = vuint_at(data, res.next).unwrap();
    assert_eq!(res.val, 0);
    assert_eq!(res.next, 16);
    res = vuint_at(data, res.next).unwrap();
    assert_eq!(res.val, (1 << 28) - 1);
    assert_eq!(res.next, 20);
}

#[bench]
pub fn vuint_at_A_aligned(b: &mut Bencher) {
    let data = (0..4 * 100)
                   .map(|i| {
                       match i % 2 {
                           0 => 0x80,
                           _ => i as u8,
                       }
                   })
                   .collect::<Vec<_>>();
    let mut sum = 0;
    b.iter(|| {
        let mut i = 0;
        while i < data.len() {
            sum += vuint_at(&data, i).unwrap().val;
            i += 4;
        }
    });
}

#[bench]
pub fn vuint_at_A_unaligned(b: &mut Bencher) {
    let data = (0..4 * 100 + 1)
                   .map(|i| {
                       match i % 2 {
                           1 => 0x80,
                           _ => i as u8,
                       }
                   })
                   .collect::<Vec<_>>();
    let mut sum = 0;
    b.iter(|| {
        let mut i = 1;
        while i < data.len() {
            sum += vuint_at(&data, i).unwrap().val;
            i += 4;
        }
    });
}

#[bench]
pub fn vuint_at_D_aligned(b: &mut Bencher) {
    let data = (0..4 * 100)
                   .map(|i| {
                       match i % 4 {
                           0 => 0x10,
                           3 => i as u8,
                           _ => 0,
                       }
                   })
                   .collect::<Vec<_>>();
    let mut sum = 0;
    b.iter(|| {
        let mut i = 0;
        while i < data.len() {
            sum += vuint_at(&data, i).unwrap().val;
            i += 4;
        }
    });
}

#[bench]
pub fn vuint_at_D_unaligned(b: &mut Bencher) {
    let data = (0..4 * 100 + 1)
                   .map(|i| {
                       match i % 4 {
                           1 => 0x10,
                           0 => i as u8,
                           _ => 0,
                       }
                   })
                   .collect::<Vec<_>>();
    let mut sum = 0;
    b.iter(|| {
        let mut i = 1;
        while i < data.len() {
            sum += vuint_at(&data, i).unwrap().val;
            i += 4;
        }
    });
}
