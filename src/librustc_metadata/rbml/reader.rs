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
//!
//! # Predefined Tags
//!
//! Most RBML tags are defined by the application.
//! (For the rust object metadata, see also `rustc::metadata::common`.)
//! RBML itself does define a set of predefined tags however,
//! intended for the auto-serialization implementation.
//!
//! Predefined tags with an implicit length:
//!
//! - `U8`  (`00`): 1-byte unsigned integer.
//! - `U16` (`01`): 2-byte big endian unsigned integer.
//! - `U32` (`02`): 4-byte big endian unsigned integer.
//! - `U64` (`03`): 8-byte big endian unsigned integer.
//!   Any of `U*` tags can be used to encode primitive unsigned integer types,
//!   as long as it is no greater than the actual size.
//!   For example, `u8` can only be represented via the `U8` tag.
//!
//! - `I8`  (`04`): 1-byte signed integer.
//! - `I16` (`05`): 2-byte big endian signed integer.
//! - `I32` (`06`): 4-byte big endian signed integer.
//! - `I64` (`07`): 8-byte big endian signed integer.
//!   Similar to `U*` tags. Always uses two's complement encoding.
//!
//! - `Bool` (`08`): 1-byte boolean value, `00` for false and `01` for true.
//!
//! - `Char` (`09`): 4-byte big endian Unicode scalar value.
//!   Surrogate pairs or out-of-bound values are invalid.
//!
//! - `F32` (`0a`): 4-byte big endian unsigned integer representing
//!   IEEE 754 binary32 floating-point format.
//! - `F64` (`0b`): 8-byte big endian unsigned integer representing
//!   IEEE 754 binary64 floating-point format.
//!
//! - `Sub8`  (`0c`): 1-byte unsigned integer for supplementary information.
//! - `Sub32` (`0d`): 4-byte unsigned integer for supplementary information.
//!   Those two tags normally occur as the first subdocument of certain tags,
//!   namely `Enum`, `Vec` and `Map`, to provide a variant or size information.
//!   They can be used interchangeably.
//!
//! Predefined tags with an explicit length:
//!
//! - `Str` (`10`): A UTF-8-encoded string.
//!
//! - `Enum` (`11`): An enum.
//!   The first subdocument should be `Sub*` tags with a variant ID.
//!   Subsequent subdocuments, if any, encode variant arguments.
//!
//! - `Vec` (`12`): A vector (sequence).
//! - `VecElt` (`13`): A vector element.
//!   The first subdocument should be `Sub*` tags with the number of elements.
//!   Subsequent subdocuments should be `VecElt` tag per each element.
//!
//! - `Map` (`14`): A map (associated array).
//! - `MapKey` (`15`): A key part of the map entry.
//! - `MapVal` (`16`): A value part of the map entry.
//!   The first subdocument should be `Sub*` tags with the number of entries.
//!   Subsequent subdocuments should be an alternating sequence of
//!   `MapKey` and `MapVal` tags per each entry.
//!
//! - `Opaque` (`17`): An opaque, custom-format tag.
//!   Used to wrap ordinary custom tags or data in the auto-serialized context.
//!   Rustc typically uses this to encode type information.
//!
//! First 0x20 tags are reserved by RBML; custom tags start at 0x20.

#[cfg(test)]
use test::Bencher;

pub use self::EbmlEncoderTag::*;

use std::char;
use std::isize;
use std::mem::transmute;
use std::str;

use rustc_serialize as serialize;

use rbml::opaque;
use rbml::Error;
use rbml::Error::*;

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

    pub fn get(&self, tag: usize) -> Doc<'doc> {
        get_doc(*self, tag)
    }

    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    pub fn as_str(&self) -> &'doc str {
        str::from_utf8(&self.data[self.start..self.end]).unwrap()
    }

    pub fn to_string(&self) -> String {
        self.as_str().to_string()
    }

    pub fn opaque(&self) -> opaque::Decoder<'doc> {
        opaque::Decoder::new(self.data, self.start)
    }
}

pub struct TaggedDoc<'a> {
    tag: usize,
    pub doc: Doc<'a>,
}

pub type DecodeResult<T> = Result<T, Error>;

#[derive(Copy, Clone, Debug)]
pub enum EbmlEncoderTag {
    // tags 00..1f are reserved for auto-serialization.
    // first NUM_IMPLICIT_TAGS tags are implicitly sized and lengths are not encoded.
    EsU8 = 0x00, // + 1 byte
    EsU16 = 0x01, // + 2 bytes
    EsU32 = 0x02, // + 4 bytes
    EsU64 = 0x03, // + 8 bytes
    EsI8 = 0x04, // + 1 byte
    EsI16 = 0x05, // + 2 bytes
    EsI32 = 0x06, // + 4 bytes
    EsI64 = 0x07, // + 8 bytes
    EsBool = 0x08, // + 1 byte
    EsChar = 0x09, // + 4 bytes
    EsF32 = 0x0a, // + 4 bytes
    EsF64 = 0x0b, // + 8 bytes
    EsSub8 = 0x0c, // + 1 byte
    EsSub32 = 0x0d, // + 4 bytes
    // 0x0e and 0x0f are reserved
    EsStr = 0x10,
    EsEnum = 0x11, // encodes the variant id as the first EsSub*
    EsVec = 0x12, // encodes the # of elements as the first EsSub*
    EsVecElt = 0x13,
    EsMap = 0x14, // encodes the # of pairs as the first EsSub*
    EsMapKey = 0x15,
    EsMapVal = 0x16,
    EsOpaque = 0x17,
}

pub const NUM_IMPLICIT_TAGS: usize = 0x0e;

#[cfg_attr(rustfmt, rustfmt_skip)]
static TAG_IMPLICIT_LEN: [i8; NUM_IMPLICIT_TAGS] = [
    1, 2, 4, 8, // EsU*
    1, 2, 4, 8, // ESI*
    1, // EsBool
    4, // EsChar
    4, 8, // EsF*
    1, 4, // EsSub*
];

// rbml reading

macro_rules! try_or {
    ($e:expr, $r:expr) => (
        match $e {
            Ok(e) => e,
            Err(e) => {
                debug!("ignored error: {:?}", e);
                return $r
            }
        }
    )
}

#[derive(Copy, Clone)]
pub struct Res {
    pub val: usize,
    pub next: usize,
}

pub fn tag_at(data: &[u8], start: usize) -> DecodeResult<Res> {
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
        Err(InvalidTag(v))
    }
}

#[inline(never)]
fn vuint_at_slow(data: &[u8], start: usize) -> DecodeResult<Res> {
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
    Err(IntTooBig(a as usize))
}

pub fn vuint_at(data: &[u8], start: usize) -> DecodeResult<Res> {
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

pub fn tag_len_at(data: &[u8], tag: Res) -> DecodeResult<Res> {
    if tag.val < NUM_IMPLICIT_TAGS && TAG_IMPLICIT_LEN[tag.val] >= 0 {
        Ok(Res {
            val: TAG_IMPLICIT_LEN[tag.val] as usize,
            next: tag.next,
        })
    } else {
        vuint_at(data, tag.next)
    }
}

pub fn doc_at<'a>(data: &'a [u8], start: usize) -> DecodeResult<TaggedDoc<'a>> {
    let elt_tag = tag_at(data, start)?;
    let elt_size = tag_len_at(data, elt_tag)?;
    let end = elt_size.next + elt_size.val;
    Ok(TaggedDoc {
        tag: elt_tag.val,
        doc: Doc {
            data: data,
            start: elt_size.next,
            end: end,
        },
    })
}

pub fn maybe_get_doc<'a>(d: Doc<'a>, tg: usize) -> Option<Doc<'a>> {
    let mut pos = d.start;
    while pos < d.end {
        let elt_tag = try_or!(tag_at(d.data, pos), None);
        let elt_size = try_or!(tag_len_at(d.data, elt_tag), None);
        pos = elt_size.next + elt_size.val;
        if elt_tag.val == tg {
            return Some(Doc {
                data: d.data,
                start: elt_size.next,
                end: pos,
            });
        }
    }
    None
}

pub fn get_doc<'a>(d: Doc<'a>, tg: usize) -> Doc<'a> {
    match maybe_get_doc(d, tg) {
        Some(d) => d,
        None => {
            error!("failed to find block with tag {:?}", tg);
            panic!();
        }
    }
}

pub fn docs<'a>(d: Doc<'a>) -> DocsIterator<'a> {
    DocsIterator { d: d }
}

pub struct DocsIterator<'a> {
    d: Doc<'a>,
}

impl<'a> Iterator for DocsIterator<'a> {
    type Item = (usize, Doc<'a>);

    fn next(&mut self) -> Option<(usize, Doc<'a>)> {
        if self.d.start >= self.d.end {
            return None;
        }

        let elt_tag = try_or!(tag_at(self.d.data, self.d.start), {
            self.d.start = self.d.end;
            None
        });
        let elt_size = try_or!(tag_len_at(self.d.data, elt_tag), {
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
        return Some((elt_tag.val, doc));
    }
}

pub fn tagged_docs<'a>(d: Doc<'a>, tag: usize) -> TaggedDocsIterator<'a> {
    TaggedDocsIterator {
        iter: docs(d),
        tag: tag,
    }
}

pub struct TaggedDocsIterator<'a> {
    iter: DocsIterator<'a>,
    tag: usize,
}

impl<'a> Iterator for TaggedDocsIterator<'a> {
    type Item = Doc<'a>;

    fn next(&mut self) -> Option<Doc<'a>> {
        while let Some((tag, doc)) = self.iter.next() {
            if tag == self.tag {
                return Some(doc);
            }
        }
        None
    }
}

pub fn with_doc_data<T, F>(d: Doc, f: F) -> T
    where F: FnOnce(&[u8]) -> T
{
    f(&d.data[d.start..d.end])
}

pub fn doc_as_u8(d: Doc) -> u8 {
    assert_eq!(d.end, d.start + 1);
    d.data[d.start]
}

pub fn doc_as_u64(d: Doc) -> u64 {
    if d.end >= 8 {
        // For performance, we read 8 big-endian bytes,
        // and mask off the junk if there is any. This
        // obviously won't work on the first 8 bytes
        // of a file - we will fall of the start
        // of the page and segfault.

        let mut b = [0; 8];
        b.copy_from_slice(&d.data[d.end - 8..d.end]);
        let data = unsafe { (*(b.as_ptr() as *const u64)).to_be() };
        let len = d.end - d.start;
        if len < 8 {
            data & ((1 << (len * 8)) - 1)
        } else {
            data
        }
    } else {
        let mut result = 0;
        for b in &d.data[d.start..d.end] {
            result = (result << 8) + (*b as u64);
        }
        result
    }
}

#[inline]
pub fn doc_as_u16(d: Doc) -> u16 {
    doc_as_u64(d) as u16
}
#[inline]
pub fn doc_as_u32(d: Doc) -> u32 {
    doc_as_u64(d) as u32
}

#[inline]
pub fn doc_as_i8(d: Doc) -> i8 {
    doc_as_u8(d) as i8
}
#[inline]
pub fn doc_as_i16(d: Doc) -> i16 {
    doc_as_u16(d) as i16
}
#[inline]
pub fn doc_as_i32(d: Doc) -> i32 {
    doc_as_u32(d) as i32
}
#[inline]
pub fn doc_as_i64(d: Doc) -> i64 {
    doc_as_u64(d) as i64
}

pub struct Decoder<'a> {
    parent: Doc<'a>,
    pos: usize,
}

impl<'doc> Decoder<'doc> {
    pub fn new(d: Doc<'doc>) -> Decoder<'doc> {
        Decoder {
            parent: d,
            pos: d.start,
        }
    }

    fn next_doc(&mut self, exp_tag: EbmlEncoderTag) -> DecodeResult<Doc<'doc>> {
        debug!(". next_doc(exp_tag={:?})", exp_tag);
        if self.pos >= self.parent.end {
            return Err(Expected(format!("no more documents in current node!")));
        }
        let TaggedDoc { tag: r_tag, doc: r_doc } = doc_at(self.parent.data, self.pos)?;
        debug!("self.parent={:?}-{:?} self.pos={:?} r_tag={:?} r_doc={:?}-{:?}",
               self.parent.start,
               self.parent.end,
               self.pos,
               r_tag,
               r_doc.start,
               r_doc.end);
        if r_tag != (exp_tag as usize) {
            return Err(Expected(format!("expected EBML doc with tag {:?} but found tag {:?}",
                                        exp_tag,
                                        r_tag)));
        }
        if r_doc.end > self.parent.end {
            return Err(Expected(format!("invalid EBML, child extends to {:#x}, parent to \
                                         {:#x}",
                                        r_doc.end,
                                        self.parent.end)));
        }
        self.pos = r_doc.end;
        Ok(r_doc)
    }

    fn push_doc<T, F>(&mut self, exp_tag: EbmlEncoderTag, f: F) -> DecodeResult<T>
        where F: FnOnce(&mut Decoder<'doc>) -> DecodeResult<T>
    {
        let d = self.next_doc(exp_tag)?;
        let old_parent = self.parent;
        let old_pos = self.pos;
        self.parent = d;
        self.pos = d.start;
        let r = f(self)?;
        self.parent = old_parent;
        self.pos = old_pos;
        Ok(r)
    }

    fn _next_sub(&mut self) -> DecodeResult<usize> {
        // empty vector/map optimization
        if self.parent.is_empty() {
            return Ok(0);
        }

        let TaggedDoc { tag: r_tag, doc: r_doc } = doc_at(self.parent.data, self.pos)?;
        let r = if r_tag == (EsSub8 as usize) {
            doc_as_u8(r_doc) as usize
        } else if r_tag == (EsSub32 as usize) {
            doc_as_u32(r_doc) as usize
        } else {
            return Err(Expected(format!("expected EBML doc with tag {:?} or {:?} but found \
                                         tag {:?}",
                                        EsSub8,
                                        EsSub32,
                                        r_tag)));
        };
        if r_doc.end > self.parent.end {
            return Err(Expected(format!("invalid EBML, child extends to {:#x}, parent to \
                                         {:#x}",
                                        r_doc.end,
                                        self.parent.end)));
        }
        self.pos = r_doc.end;
        debug!("_next_sub result={:?}", r);
        Ok(r)
    }

    // variable-length unsigned integer with different tags.
    // `last_tag` should be the largest allowed unsigned integer tag.
    // all tags between them should be valid, in the order of u8, u16, u32 and u64.
    fn next_uint(&mut self,
                 last_tag: EbmlEncoderTag)
                 -> DecodeResult<u64> {
        if self.pos >= self.parent.end {
            return Err(Expected(format!("no more documents in current node!")));
        }

        let TaggedDoc { tag: r_tag, doc: r_doc } = doc_at(self.parent.data, self.pos)?;
        let r = if EsU8 as usize <= r_tag && r_tag <= last_tag as usize {
            match r_tag - EsU8 as usize {
                0 => doc_as_u8(r_doc) as u64,
                1 => doc_as_u16(r_doc) as u64,
                2 => doc_as_u32(r_doc) as u64,
                3 => doc_as_u64(r_doc),
                _ => unreachable!(),
            }
        } else {
            return Err(Expected(format!("expected EBML doc with tag EsU8 through {:?} but \
                                         found tag {:?}",
                                        last_tag,
                                        r_tag)));
        };
        if r_doc.end > self.parent.end {
            return Err(Expected(format!("invalid EBML, child extends to {:#x}, parent to \
                                         {:#x}",
                                        r_doc.end,
                                        self.parent.end)));
        }
        self.pos = r_doc.end;
        debug!("next_uint({:?}) result={:?}", last_tag, r);
        Ok(r)
    }

    // variable-length signed integer with different tags.
    // `last_tag` should be the largest allowed signed integer tag.
    // all tags between them should be valid, in the order of i8, i16, i32 and i64.
    fn next_int(&mut self,
                last_tag: EbmlEncoderTag)
                -> DecodeResult<i64> {
        if self.pos >= self.parent.end {
            return Err(Expected(format!("no more documents in current node!")));
        }

        let TaggedDoc { tag: r_tag, doc: r_doc } = doc_at(self.parent.data, self.pos)?;
        let r = if EsI8 as usize <= r_tag && r_tag <= last_tag as usize {
            match r_tag - EsI8 as usize {
                0 => doc_as_i8(r_doc) as i64,
                1 => doc_as_i16(r_doc) as i64,
                2 => doc_as_i32(r_doc) as i64,
                3 => doc_as_i64(r_doc),
                _ => unreachable!(),
            }
        } else {
            return Err(Expected(format!("expected EBML doc with tag EsI8 through {:?} but \
                                         found tag {:?}",
                                        last_tag,
                                        r_tag)));
        };
        if r_doc.end > self.parent.end {
            return Err(Expected(format!("invalid EBML, child extends to {:#x}, parent to \
                                         {:#x}",
                                        r_doc.end,
                                        self.parent.end)));
        }
        self.pos = r_doc.end;
        debug!("next_int({:?}) result={:?}", last_tag, r);
        Ok(r)
    }

    pub fn read_opaque<R, F>(&mut self, op: F) -> DecodeResult<R>
        where F: FnOnce(&mut opaque::Decoder, Doc) -> DecodeResult<R>
    {
        let doc = self.next_doc(EsOpaque)?;
        let result = op(&mut doc.opaque(), doc)?;
        Ok(result)
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    pub fn advance(&mut self, bytes: usize) {
        self.pos += bytes;
    }
}

impl<'doc> serialize::Decoder for Decoder<'doc> {
    type Error = Error;
    fn read_nil(&mut self) -> DecodeResult<()> {
        Ok(())
    }

    fn read_u64(&mut self) -> DecodeResult<u64> {
        self.next_uint(EsU64)
    }
    fn read_u32(&mut self) -> DecodeResult<u32> {
        Ok(self.next_uint(EsU32)? as u32)
    }
    fn read_u16(&mut self) -> DecodeResult<u16> {
        Ok(self.next_uint(EsU16)? as u16)
    }
    fn read_u8(&mut self) -> DecodeResult<u8> {
        Ok(doc_as_u8(self.next_doc(EsU8)?))
    }
    fn read_usize(&mut self) -> DecodeResult<usize> {
        let v = self.read_u64()?;
        if v > (::std::usize::MAX as u64) {
            Err(IntTooBig(v as usize))
        } else {
            Ok(v as usize)
        }
    }

    fn read_i64(&mut self) -> DecodeResult<i64> {
        Ok(self.next_int(EsI64)? as i64)
    }
    fn read_i32(&mut self) -> DecodeResult<i32> {
        Ok(self.next_int(EsI32)? as i32)
    }
    fn read_i16(&mut self) -> DecodeResult<i16> {
        Ok(self.next_int(EsI16)? as i16)
    }
    fn read_i8(&mut self) -> DecodeResult<i8> {
        Ok(doc_as_u8(self.next_doc(EsI8)?) as i8)
    }
    fn read_isize(&mut self) -> DecodeResult<isize> {
        let v = self.next_int(EsI64)? as i64;
        if v > (isize::MAX as i64) || v < (isize::MIN as i64) {
            debug!("FIXME \\#6122: Removing this makes this function miscompile");
            Err(IntTooBig(v as usize))
        } else {
            Ok(v as isize)
        }
    }

    fn read_bool(&mut self) -> DecodeResult<bool> {
        Ok(doc_as_u8(self.next_doc(EsBool)?) != 0)
    }

    fn read_f64(&mut self) -> DecodeResult<f64> {
        let bits = doc_as_u64(self.next_doc(EsF64)?);
        Ok(unsafe { transmute(bits) })
    }
    fn read_f32(&mut self) -> DecodeResult<f32> {
        let bits = doc_as_u32(self.next_doc(EsF32)?);
        Ok(unsafe { transmute(bits) })
    }
    fn read_char(&mut self) -> DecodeResult<char> {
        Ok(char::from_u32(doc_as_u32(self.next_doc(EsChar)?)).unwrap())
    }
    fn read_str(&mut self) -> DecodeResult<String> {
        Ok(self.next_doc(EsStr)?.to_string())
    }

    // Compound types:
    fn read_enum<T, F>(&mut self, name: &str, f: F) -> DecodeResult<T>
        where F: FnOnce(&mut Decoder<'doc>) -> DecodeResult<T>
    {
        debug!("read_enum({})", name);

        let doc = self.next_doc(EsEnum)?;

        let (old_parent, old_pos) = (self.parent, self.pos);
        self.parent = doc;
        self.pos = self.parent.start;

        let result = f(self)?;

        self.parent = old_parent;
        self.pos = old_pos;
        Ok(result)
    }

    fn read_enum_variant<T, F>(&mut self, _: &[&str], mut f: F) -> DecodeResult<T>
        where F: FnMut(&mut Decoder<'doc>, usize) -> DecodeResult<T>
    {
        debug!("read_enum_variant()");
        let idx = self._next_sub()?;
        debug!("  idx={}", idx);

        f(self, idx)
    }

    fn read_enum_variant_arg<T, F>(&mut self, idx: usize, f: F) -> DecodeResult<T>
        where F: FnOnce(&mut Decoder<'doc>) -> DecodeResult<T>
    {
        debug!("read_enum_variant_arg(idx={})", idx);
        f(self)
    }

    fn read_enum_struct_variant<T, F>(&mut self, _: &[&str], mut f: F) -> DecodeResult<T>
        where F: FnMut(&mut Decoder<'doc>, usize) -> DecodeResult<T>
    {
        debug!("read_enum_struct_variant()");
        let idx = self._next_sub()?;
        debug!("  idx={}", idx);

        f(self, idx)
    }

    fn read_enum_struct_variant_field<T, F>(&mut self,
                                            name: &str,
                                            idx: usize,
                                            f: F)
                                            -> DecodeResult<T>
        where F: FnOnce(&mut Decoder<'doc>) -> DecodeResult<T>
    {
        debug!("read_enum_struct_variant_arg(name={}, idx={})", name, idx);
        f(self)
    }

    fn read_struct<T, F>(&mut self, name: &str, _: usize, f: F) -> DecodeResult<T>
        where F: FnOnce(&mut Decoder<'doc>) -> DecodeResult<T>
    {
        debug!("read_struct(name={})", name);
        f(self)
    }

    fn read_struct_field<T, F>(&mut self, name: &str, idx: usize, f: F) -> DecodeResult<T>
        where F: FnOnce(&mut Decoder<'doc>) -> DecodeResult<T>
    {
        debug!("read_struct_field(name={}, idx={})", name, idx);
        f(self)
    }

    fn read_tuple<T, F>(&mut self, tuple_len: usize, f: F) -> DecodeResult<T>
        where F: FnOnce(&mut Decoder<'doc>) -> DecodeResult<T>
    {
        debug!("read_tuple()");
        self.read_seq(move |d, len| {
            if len == tuple_len {
                f(d)
            } else {
                Err(Expected(format!("Expected tuple of length `{}`, found tuple of length \
                                      `{}`",
                                     tuple_len,
                                     len)))
            }
        })
    }

    fn read_tuple_arg<T, F>(&mut self, idx: usize, f: F) -> DecodeResult<T>
        where F: FnOnce(&mut Decoder<'doc>) -> DecodeResult<T>
    {
        debug!("read_tuple_arg(idx={})", idx);
        self.read_seq_elt(idx, f)
    }

    fn read_tuple_struct<T, F>(&mut self, name: &str, len: usize, f: F) -> DecodeResult<T>
        where F: FnOnce(&mut Decoder<'doc>) -> DecodeResult<T>
    {
        debug!("read_tuple_struct(name={})", name);
        self.read_tuple(len, f)
    }

    fn read_tuple_struct_arg<T, F>(&mut self, idx: usize, f: F) -> DecodeResult<T>
        where F: FnOnce(&mut Decoder<'doc>) -> DecodeResult<T>
    {
        debug!("read_tuple_struct_arg(idx={})", idx);
        self.read_tuple_arg(idx, f)
    }

    fn read_option<T, F>(&mut self, mut f: F) -> DecodeResult<T>
        where F: FnMut(&mut Decoder<'doc>, bool) -> DecodeResult<T>
    {
        debug!("read_option()");
        self.read_enum("Option", move |this| {
            this.read_enum_variant(&["None", "Some"], move |this, idx| {
                match idx {
                    0 => f(this, false),
                    1 => f(this, true),
                    _ => Err(Expected(format!("Expected None or Some"))),
                }
            })
        })
    }

    fn read_seq<T, F>(&mut self, f: F) -> DecodeResult<T>
        where F: FnOnce(&mut Decoder<'doc>, usize) -> DecodeResult<T>
    {
        debug!("read_seq()");
        self.push_doc(EsVec, move |d| {
            let len = d._next_sub()?;
            debug!("  len={}", len);
            f(d, len)
        })
    }

    fn read_seq_elt<T, F>(&mut self, idx: usize, f: F) -> DecodeResult<T>
        where F: FnOnce(&mut Decoder<'doc>) -> DecodeResult<T>
    {
        debug!("read_seq_elt(idx={})", idx);
        self.push_doc(EsVecElt, f)
    }

    fn read_map<T, F>(&mut self, f: F) -> DecodeResult<T>
        where F: FnOnce(&mut Decoder<'doc>, usize) -> DecodeResult<T>
    {
        debug!("read_map()");
        self.push_doc(EsMap, move |d| {
            let len = d._next_sub()?;
            debug!("  len={}", len);
            f(d, len)
        })
    }

    fn read_map_elt_key<T, F>(&mut self, idx: usize, f: F) -> DecodeResult<T>
        where F: FnOnce(&mut Decoder<'doc>) -> DecodeResult<T>
    {
        debug!("read_map_elt_key(idx={})", idx);
        self.push_doc(EsMapKey, f)
    }

    fn read_map_elt_val<T, F>(&mut self, idx: usize, f: F) -> DecodeResult<T>
        where F: FnOnce(&mut Decoder<'doc>) -> DecodeResult<T>
    {
        debug!("read_map_elt_val(idx={})", idx);
        self.push_doc(EsMapVal, f)
    }

    fn error(&mut self, err: &str) -> Error {
        ApplicationError(err.to_string())
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
