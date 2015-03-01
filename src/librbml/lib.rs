// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Really Bad Markup Language (rbml) is a temporary measure until we migrate
//! the rust object metadata to a better serialization format. It is not
//! intended to be used by users.
//!
//! It is loosely based on the Extensible Binary Markup Language (ebml):
//!     http://www.matroska.org/technical/specs/rfc/index.html

#![crate_name = "rbml"]
#![unstable(feature = "rustc_private")]
#![staged_api]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/nightly/",
       html_playground_url = "http://play.rust-lang.org/")]

#![feature(collections)]
#![feature(core)]
#![feature(int_uint)]
#![feature(old_io)]
#![feature(rustc_private)]
#![feature(staged_api)]

#![cfg_attr(test, feature(test))]

extern crate serialize;
#[macro_use] extern crate log;

#[cfg(test)] extern crate test;

pub use self::EbmlEncoderTag::*;
pub use self::Error::*;

use std::str;
use std::fmt;

pub mod io;

/// Common data structures
#[derive(Clone, Copy)]
pub struct Doc<'a> {
    pub data: &'a [u8],
    pub start: uint,
    pub end: uint,
}

impl<'doc> Doc<'doc> {
    pub fn new(data: &'doc [u8]) -> Doc<'doc> {
        Doc { data: data, start: 0, end: data.len() }
    }

    pub fn get<'a>(&'a self, tag: uint) -> Doc<'a> {
        reader::get_doc(*self, tag)
    }

    pub fn as_str_slice<'a>(&'a self) -> &'a str {
        str::from_utf8(&self.data[self.start..self.end]).unwrap()
    }

    pub fn as_str(&self) -> String {
        self.as_str_slice().to_string()
    }
}

pub struct TaggedDoc<'a> {
    tag: uint,
    pub doc: Doc<'a>,
}

#[derive(Copy, Debug)]
pub enum EbmlEncoderTag {
    // tags 00..1f are reserved for auto-serialization.
    // first NUM_IMPLICIT_TAGS tags are implicitly sized and lengths are not encoded.

    EsUint     = 0x00, // + 8 bytes
    EsU64      = 0x01, // + 8 bytes
    EsU32      = 0x02, // + 4 bytes
    EsU16      = 0x03, // + 2 bytes
    EsU8       = 0x04, // + 1 byte
    EsInt      = 0x05, // + 8 bytes
    EsI64      = 0x06, // + 8 bytes
    EsI32      = 0x07, // + 4 bytes
    EsI16      = 0x08, // + 2 bytes
    EsI8       = 0x09, // + 1 byte
    EsBool     = 0x0a, // + 1 byte
    EsChar     = 0x0b, // + 4 bytes
    EsF64      = 0x0c, // + 8 bytes
    EsF32      = 0x0d, // + 4 bytes
    EsSub8     = 0x0e, // + 1 byte
    EsSub32    = 0x0f, // + 4 bytes

    EsStr      = 0x10,
    EsEnum     = 0x11, // encodes the variant id as the first EsSub*
    EsVec      = 0x12, // encodes the # of elements as the first EsSub*
    EsVecElt   = 0x13,
    EsMap      = 0x14, // encodes the # of pairs as the first EsSub*
    EsMapKey   = 0x15,
    EsMapVal   = 0x16,
    EsOpaque   = 0x17,
}

const NUM_TAGS: uint = 0x1000;
const NUM_IMPLICIT_TAGS: uint = 0x10;

static TAG_IMPLICIT_LEN: [i8; NUM_IMPLICIT_TAGS] = [
    8, 8, 4, 2, 1, // EsU*
    8, 8, 4, 2, 1, // ESI*
    1, // EsBool
    4, // EsChar
    8, 4, // EsF*
    1, 4, // EsSub*
];

#[derive(Debug)]
pub enum Error {
    IntTooBig(uint),
    InvalidTag(uint),
    Expected(String),
    IoError(std::old_io::IoError),
    ApplicationError(String)
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // FIXME: this should be a more useful display form
        fmt::Debug::fmt(self, f)
    }
}
// --------------------------------------

pub mod reader {
    use std::char;

    use std::isize;
    use std::mem::transmute;
    use std::num::Int;
    use std::slice::bytes;

    use serialize;

    use super::{ ApplicationError, EsVec, EsMap, EsEnum, EsSub8, EsSub32,
        EsVecElt, EsMapKey, EsU64, EsU32, EsU16, EsU8, EsInt, EsI64,
        EsI32, EsI16, EsI8, EsBool, EsF64, EsF32, EsChar, EsStr, EsMapVal,
        EsUint, EsOpaque, EbmlEncoderTag, Doc, TaggedDoc,
        Error, IntTooBig, InvalidTag, Expected, NUM_IMPLICIT_TAGS, TAG_IMPLICIT_LEN };

    pub type DecodeResult<T> = Result<T, Error>;
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

    #[derive(Copy)]
    pub struct Res {
        pub val: uint,
        pub next: uint
    }

    pub fn tag_at(data: &[u8], start: uint) -> DecodeResult<Res> {
        let v = data[start] as uint;
        if v < 0xf0 {
            Ok(Res { val: v, next: start + 1 })
        } else if v > 0xf0 {
            Ok(Res { val: ((v & 0xf) << 8) | data[start + 1] as uint, next: start + 2 })
        } else {
            // every tag starting with byte 0xf0 is an overlong form, which is prohibited.
            Err(InvalidTag(v))
        }
    }

    #[inline(never)]
    fn vuint_at_slow(data: &[u8], start: uint) -> DecodeResult<Res> {
        let a = data[start];
        if a & 0x80u8 != 0u8 {
            return Ok(Res {val: (a & 0x7fu8) as uint, next: start + 1});
        }
        if a & 0x40u8 != 0u8 {
            return Ok(Res {val: ((a & 0x3fu8) as uint) << 8 |
                        (data[start + 1] as uint),
                    next: start + 2});
        }
        if a & 0x20u8 != 0u8 {
            return Ok(Res {val: ((a & 0x1fu8) as uint) << 16 |
                        (data[start + 1] as uint) << 8 |
                        (data[start + 2] as uint),
                    next: start + 3});
        }
        if a & 0x10u8 != 0u8 {
            return Ok(Res {val: ((a & 0x0fu8) as uint) << 24 |
                        (data[start + 1] as uint) << 16 |
                        (data[start + 2] as uint) << 8 |
                        (data[start + 3] as uint),
                    next: start + 4});
        }
        Err(IntTooBig(a as uint))
    }

    pub fn vuint_at(data: &[u8], start: uint) -> DecodeResult<Res> {
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
        static SHIFT_MASK_TABLE: [(uint, u32); 16] = [
            (0, 0x0), (0, 0x0fffffff),
            (8, 0x1fffff), (8, 0x1fffff),
            (16, 0x3fff), (16, 0x3fff), (16, 0x3fff), (16, 0x3fff),
            (24, 0x7f), (24, 0x7f), (24, 0x7f), (24, 0x7f),
            (24, 0x7f), (24, 0x7f), (24, 0x7f), (24, 0x7f)
        ];

        unsafe {
            let ptr = data.as_ptr().offset(start as int) as *const u32;
            let val = Int::from_be(*ptr);

            let i = (val >> 28) as uint;
            let (shift, mask) = SHIFT_MASK_TABLE[i];
            Ok(Res {
                val: ((val >> shift) & mask) as uint,
                next: start + (((32 - shift) >> 3) as uint)
            })
        }
    }

    pub fn tag_len_at(data: &[u8], tag: Res) -> DecodeResult<Res> {
        if tag.val < NUM_IMPLICIT_TAGS && TAG_IMPLICIT_LEN[tag.val] >= 0 {
            Ok(Res { val: TAG_IMPLICIT_LEN[tag.val] as uint, next: tag.next })
        } else {
            vuint_at(data, tag.next)
        }
    }

    pub fn doc_at<'a>(data: &'a [u8], start: uint) -> DecodeResult<TaggedDoc<'a>> {
        let elt_tag = try!(tag_at(data, start));
        let elt_size = try!(tag_len_at(data, elt_tag));
        let end = elt_size.next + elt_size.val;
        Ok(TaggedDoc {
            tag: elt_tag.val,
            doc: Doc { data: data, start: elt_size.next, end: end }
        })
    }

    pub fn maybe_get_doc<'a>(d: Doc<'a>, tg: uint) -> Option<Doc<'a>> {
        let mut pos = d.start;
        while pos < d.end {
            let elt_tag = try_or!(tag_at(d.data, pos), None);
            let elt_size = try_or!(tag_len_at(d.data, elt_tag), None);
            pos = elt_size.next + elt_size.val;
            if elt_tag.val == tg {
                return Some(Doc { data: d.data, start: elt_size.next,
                                  end: pos });
            }
        }
        None
    }

    pub fn get_doc<'a>(d: Doc<'a>, tg: uint) -> Doc<'a> {
        match maybe_get_doc(d, tg) {
            Some(d) => d,
            None => {
                error!("failed to find block with tag {:?}", tg);
                panic!();
            }
        }
    }

    pub fn docs<F>(d: Doc, mut it: F) -> bool where
        F: FnMut(uint, Doc) -> bool,
    {
        let mut pos = d.start;
        while pos < d.end {
            let elt_tag = try_or!(tag_at(d.data, pos), false);
            let elt_size = try_or!(tag_len_at(d.data, elt_tag), false);
            pos = elt_size.next + elt_size.val;
            let doc = Doc { data: d.data, start: elt_size.next, end: pos };
            if !it(elt_tag.val, doc) {
                return false;
            }
        }
        return true;
    }

    pub fn tagged_docs<F>(d: Doc, tg: uint, mut it: F) -> bool where
        F: FnMut(Doc) -> bool,
    {
        let mut pos = d.start;
        while pos < d.end {
            let elt_tag = try_or!(tag_at(d.data, pos), false);
            let elt_size = try_or!(tag_len_at(d.data, elt_tag), false);
            pos = elt_size.next + elt_size.val;
            if elt_tag.val == tg {
                let doc = Doc { data: d.data, start: elt_size.next,
                                end: pos };
                if !it(doc) {
                    return false;
                }
            }
        }
        return true;
    }

    pub fn with_doc_data<T, F>(d: Doc, f: F) -> T where
        F: FnOnce(&[u8]) -> T,
    {
        f(&d.data[d.start..d.end])
    }


    pub fn doc_as_u8(d: Doc) -> u8 {
        assert_eq!(d.end, d.start + 1);
        d.data[d.start]
    }

    pub fn doc_as_u16(d: Doc) -> u16 {
        assert_eq!(d.end, d.start + 2);
        let mut b = [0; 2];
        bytes::copy_memory(&mut b, &d.data[d.start..d.end]);
        unsafe { (*(b.as_ptr() as *const u16)).to_be() }
    }

    pub fn doc_as_u32(d: Doc) -> u32 {
        assert_eq!(d.end, d.start + 4);
        let mut b = [0; 4];
        bytes::copy_memory(&mut b, &d.data[d.start..d.end]);
        unsafe { (*(b.as_ptr() as *const u32)).to_be() }
    }

    pub fn doc_as_u64(d: Doc) -> u64 {
        assert_eq!(d.end, d.start + 8);
        let mut b = [0; 8];
        bytes::copy_memory(&mut b, &d.data[d.start..d.end]);
        unsafe { (*(b.as_ptr() as *const u64)).to_be() }
    }

    pub fn doc_as_i8(d: Doc) -> i8 { doc_as_u8(d) as i8 }
    pub fn doc_as_i16(d: Doc) -> i16 { doc_as_u16(d) as i16 }
    pub fn doc_as_i32(d: Doc) -> i32 { doc_as_u32(d) as i32 }
    pub fn doc_as_i64(d: Doc) -> i64 { doc_as_u64(d) as i64 }

    pub struct Decoder<'a> {
        parent: Doc<'a>,
        pos: uint,
    }

    impl<'doc> Decoder<'doc> {
        pub fn new(d: Doc<'doc>) -> Decoder<'doc> {
            Decoder {
                parent: d,
                pos: d.start
            }
        }

        fn next_doc(&mut self, exp_tag: EbmlEncoderTag) -> DecodeResult<Doc<'doc>> {
            debug!(". next_doc(exp_tag={:?})", exp_tag);
            if self.pos >= self.parent.end {
                return Err(Expected(format!("no more documents in \
                                             current node!")));
            }
            let TaggedDoc { tag: r_tag, doc: r_doc } =
                try!(doc_at(self.parent.data, self.pos));
            debug!("self.parent={:?}-{:?} self.pos={:?} r_tag={:?} r_doc={:?}-{:?}",
                   self.parent.start,
                   self.parent.end,
                   self.pos,
                   r_tag,
                   r_doc.start,
                   r_doc.end);
            if r_tag != (exp_tag as uint) {
                return Err(Expected(format!("expected EBML doc with tag {:?} but \
                                             found tag {:?}", exp_tag, r_tag)));
            }
            if r_doc.end > self.parent.end {
                return Err(Expected(format!("invalid EBML, child extends to \
                                             {:#x}, parent to {:#x}",
                                            r_doc.end, self.parent.end)));
            }
            self.pos = r_doc.end;
            Ok(r_doc)
        }

        fn next_doc2(&mut self,
                     exp_tag1: EbmlEncoderTag,
                     exp_tag2: EbmlEncoderTag) -> DecodeResult<(bool, Doc<'doc>)> {
            assert!((exp_tag1 as uint) != (exp_tag2 as uint));
            debug!(". next_doc2(exp_tag1={:?}, exp_tag2={:?})", exp_tag1, exp_tag2);
            if self.pos >= self.parent.end {
                return Err(Expected(format!("no more documents in \
                                             current node!")));
            }
            let TaggedDoc { tag: r_tag, doc: r_doc } =
                try!(doc_at(self.parent.data, self.pos));
            debug!("self.parent={:?}-{:?} self.pos={:?} r_tag={:?} r_doc={:?}-{:?}",
                   self.parent.start,
                   self.parent.end,
                   self.pos,
                   r_tag,
                   r_doc.start,
                   r_doc.end);
            if r_tag != (exp_tag1 as uint) && r_tag != (exp_tag2 as uint) {
                return Err(Expected(format!("expected EBML doc with tag {:?} or {:?} but \
                                             found tag {:?}", exp_tag1, exp_tag2, r_tag)));
            }
            if r_doc.end > self.parent.end {
                return Err(Expected(format!("invalid EBML, child extends to \
                                             {:#x}, parent to {:#x}",
                                            r_doc.end, self.parent.end)));
            }
            self.pos = r_doc.end;
            Ok((r_tag == (exp_tag2 as uint), r_doc))
        }

        fn push_doc<T, F>(&mut self, exp_tag: EbmlEncoderTag, f: F) -> DecodeResult<T> where
            F: FnOnce(&mut Decoder<'doc>) -> DecodeResult<T>,
        {
            let d = try!(self.next_doc(exp_tag));
            let old_parent = self.parent;
            let old_pos = self.pos;
            self.parent = d;
            self.pos = d.start;
            let r = try!(f(self));
            self.parent = old_parent;
            self.pos = old_pos;
            Ok(r)
        }

        fn _next_sub(&mut self) -> DecodeResult<uint> {
            let (big, doc) = try!(self.next_doc2(EsSub8, EsSub32));
            let r = if big {
                doc_as_u32(doc) as uint
            } else {
                doc_as_u8(doc) as uint
            };
            debug!("_next_sub result={:?}", r);
            Ok(r)
        }

        pub fn read_opaque<R, F>(&mut self, op: F) -> DecodeResult<R> where
            F: FnOnce(&mut Decoder, Doc) -> DecodeResult<R>,
        {
            let doc = try!(self.next_doc(EsOpaque));

            let (old_parent, old_pos) = (self.parent, self.pos);
            self.parent = doc;
            self.pos = doc.start;

            let result = try!(op(self, doc));

            self.parent = old_parent;
            self.pos = old_pos;
            Ok(result)
        }
    }

    impl<'doc> serialize::Decoder for Decoder<'doc> {
        type Error = Error;
        fn read_nil(&mut self) -> DecodeResult<()> { Ok(()) }

        fn read_u64(&mut self) -> DecodeResult<u64> { Ok(doc_as_u64(try!(self.next_doc(EsU64)))) }
        fn read_u32(&mut self) -> DecodeResult<u32> { Ok(doc_as_u32(try!(self.next_doc(EsU32)))) }
        fn read_u16(&mut self) -> DecodeResult<u16> { Ok(doc_as_u16(try!(self.next_doc(EsU16)))) }
        fn read_u8 (&mut self) -> DecodeResult<u8 > { Ok(doc_as_u8 (try!(self.next_doc(EsU8 )))) }
        fn read_uint(&mut self) -> DecodeResult<uint> {
            let v = doc_as_u64(try!(self.next_doc(EsUint)));
            if v > (::std::usize::MAX as u64) {
                Err(IntTooBig(v as uint))
            } else {
                Ok(v as uint)
            }
        }

        fn read_i64(&mut self) -> DecodeResult<i64> {
            Ok(doc_as_u64(try!(self.next_doc(EsI64))) as i64)
        }
        fn read_i32(&mut self) -> DecodeResult<i32> {
            Ok(doc_as_u32(try!(self.next_doc(EsI32))) as i32)
        }
        fn read_i16(&mut self) -> DecodeResult<i16> {
            Ok(doc_as_u16(try!(self.next_doc(EsI16))) as i16)
        }
        fn read_i8 (&mut self) -> DecodeResult<i8> {
            Ok(doc_as_u8(try!(self.next_doc(EsI8 ))) as i8)
        }
        fn read_int(&mut self) -> DecodeResult<int> {
            let v = doc_as_u64(try!(self.next_doc(EsInt))) as i64;
            if v > (isize::MAX as i64) || v < (isize::MIN as i64) {
                debug!("FIXME \\#6122: Removing this makes this function miscompile");
                Err(IntTooBig(v as uint))
            } else {
                Ok(v as int)
            }
        }

        fn read_bool(&mut self) -> DecodeResult<bool> {
            Ok(doc_as_u8(try!(self.next_doc(EsBool))) != 0)
        }

        fn read_f64(&mut self) -> DecodeResult<f64> {
            let bits = doc_as_u64(try!(self.next_doc(EsF64)));
            Ok(unsafe { transmute(bits) })
        }
        fn read_f32(&mut self) -> DecodeResult<f32> {
            let bits = doc_as_u32(try!(self.next_doc(EsF32)));
            Ok(unsafe { transmute(bits) })
        }
        fn read_char(&mut self) -> DecodeResult<char> {
            Ok(char::from_u32(doc_as_u32(try!(self.next_doc(EsChar)))).unwrap())
        }
        fn read_str(&mut self) -> DecodeResult<String> {
            Ok(try!(self.next_doc(EsStr)).as_str())
        }

        // Compound types:
        fn read_enum<T, F>(&mut self, name: &str, f: F) -> DecodeResult<T> where
            F: FnOnce(&mut Decoder<'doc>) -> DecodeResult<T>,
        {
            debug!("read_enum({})", name);

            let doc = try!(self.next_doc(EsEnum));

            let (old_parent, old_pos) = (self.parent, self.pos);
            self.parent = doc;
            self.pos = self.parent.start;

            let result = try!(f(self));

            self.parent = old_parent;
            self.pos = old_pos;
            Ok(result)
        }

        fn read_enum_variant<T, F>(&mut self, _: &[&str],
                                   mut f: F) -> DecodeResult<T>
            where F: FnMut(&mut Decoder<'doc>, uint) -> DecodeResult<T>,
        {
            debug!("read_enum_variant()");
            let idx = try!(self._next_sub());
            debug!("  idx={}", idx);

            f(self, idx)
        }

        fn read_enum_variant_arg<T, F>(&mut self, idx: uint, f: F) -> DecodeResult<T> where
            F: FnOnce(&mut Decoder<'doc>) -> DecodeResult<T>,
        {
            debug!("read_enum_variant_arg(idx={})", idx);
            f(self)
        }

        fn read_enum_struct_variant<T, F>(&mut self, _: &[&str],
                                          mut f: F) -> DecodeResult<T>
            where F: FnMut(&mut Decoder<'doc>, uint) -> DecodeResult<T>,
        {
            debug!("read_enum_struct_variant()");
            let idx = try!(self._next_sub());
            debug!("  idx={}", idx);

            f(self, idx)
        }

        fn read_enum_struct_variant_field<T, F>(&mut self,
                                                name: &str,
                                                idx: uint,
                                                f: F)
                                                -> DecodeResult<T> where
            F: FnOnce(&mut Decoder<'doc>) -> DecodeResult<T>,
        {
                debug!("read_enum_struct_variant_arg(name={}, idx={})", name, idx);
            f(self)
        }

        fn read_struct<T, F>(&mut self, name: &str, _: uint, f: F) -> DecodeResult<T> where
            F: FnOnce(&mut Decoder<'doc>) -> DecodeResult<T>,
        {
            debug!("read_struct(name={})", name);
            f(self)
        }

        fn read_struct_field<T, F>(&mut self, name: &str, idx: uint, f: F) -> DecodeResult<T> where
            F: FnOnce(&mut Decoder<'doc>) -> DecodeResult<T>,
        {
            debug!("read_struct_field(name={}, idx={})", name, idx);
            f(self)
        }

        fn read_tuple<T, F>(&mut self, tuple_len: uint, f: F) -> DecodeResult<T> where
            F: FnOnce(&mut Decoder<'doc>) -> DecodeResult<T>,
        {
            debug!("read_tuple()");
            self.read_seq(move |d, len| {
                if len == tuple_len {
                    f(d)
                } else {
                    Err(Expected(format!("Expected tuple of length `{}`, \
                                          found tuple of length `{}`", tuple_len, len)))
                }
            })
        }

        fn read_tuple_arg<T, F>(&mut self, idx: uint, f: F) -> DecodeResult<T> where
            F: FnOnce(&mut Decoder<'doc>) -> DecodeResult<T>,
        {
            debug!("read_tuple_arg(idx={})", idx);
            self.read_seq_elt(idx, f)
        }

        fn read_tuple_struct<T, F>(&mut self, name: &str, len: uint, f: F) -> DecodeResult<T> where
            F: FnOnce(&mut Decoder<'doc>) -> DecodeResult<T>,
        {
            debug!("read_tuple_struct(name={})", name);
            self.read_tuple(len, f)
        }

        fn read_tuple_struct_arg<T, F>(&mut self,
                                       idx: uint,
                                       f: F)
                                       -> DecodeResult<T> where
            F: FnOnce(&mut Decoder<'doc>) -> DecodeResult<T>,
        {
            debug!("read_tuple_struct_arg(idx={})", idx);
            self.read_tuple_arg(idx, f)
        }

        fn read_option<T, F>(&mut self, mut f: F) -> DecodeResult<T> where
            F: FnMut(&mut Decoder<'doc>, bool) -> DecodeResult<T>,
        {
            debug!("read_option()");
            self.read_enum("Option", move |this| {
                this.read_enum_variant(&["None", "Some"], move |this, idx| {
                    match idx {
                        0 => f(this, false),
                        1 => f(this, true),
                        _ => {
                            Err(Expected(format!("Expected None or Some")))
                        }
                    }
                })
            })
        }

        fn read_seq<T, F>(&mut self, f: F) -> DecodeResult<T> where
            F: FnOnce(&mut Decoder<'doc>, uint) -> DecodeResult<T>,
        {
            debug!("read_seq()");
            self.push_doc(EsVec, move |d| {
                let len = try!(d._next_sub());
                debug!("  len={}", len);
                f(d, len)
            })
        }

        fn read_seq_elt<T, F>(&mut self, idx: uint, f: F) -> DecodeResult<T> where
            F: FnOnce(&mut Decoder<'doc>) -> DecodeResult<T>,
        {
            debug!("read_seq_elt(idx={})", idx);
            self.push_doc(EsVecElt, f)
        }

        fn read_map<T, F>(&mut self, f: F) -> DecodeResult<T> where
            F: FnOnce(&mut Decoder<'doc>, uint) -> DecodeResult<T>,
        {
            debug!("read_map()");
            self.push_doc(EsMap, move |d| {
                let len = try!(d._next_sub());
                debug!("  len={}", len);
                f(d, len)
            })
        }

        fn read_map_elt_key<T, F>(&mut self, idx: uint, f: F) -> DecodeResult<T> where
            F: FnOnce(&mut Decoder<'doc>) -> DecodeResult<T>,
        {
            debug!("read_map_elt_key(idx={})", idx);
            self.push_doc(EsMapKey, f)
        }

        fn read_map_elt_val<T, F>(&mut self, idx: uint, f: F) -> DecodeResult<T> where
            F: FnOnce(&mut Decoder<'doc>) -> DecodeResult<T>,
        {
            debug!("read_map_elt_val(idx={})", idx);
            self.push_doc(EsMapVal, f)
        }

        fn error(&mut self, err: &str) -> Error {
            ApplicationError(err.to_string())
        }
    }
}

pub mod writer {
    use std::mem;
    use std::num::Int;
    use std::old_io::{Writer, Seek};
    use std::old_io;

    use super::{ EsVec, EsMap, EsEnum, EsSub8, EsSub32, EsVecElt, EsMapKey,
        EsU64, EsU32, EsU16, EsU8, EsInt, EsI64, EsI32, EsI16, EsI8,
        EsBool, EsF64, EsF32, EsChar, EsStr, EsMapVal, EsUint,
        EsOpaque, NUM_IMPLICIT_TAGS, NUM_TAGS };

    use serialize;


    pub type EncodeResult = old_io::IoResult<()>;

    // rbml writing
    pub struct Encoder<'a, W:'a> {
        pub writer: &'a mut W,
        size_positions: Vec<uint>,
    }

    fn write_tag<W: Writer>(w: &mut W, n: uint) -> EncodeResult {
        if n < 0xf0 {
            w.write_all(&[n as u8])
        } else if 0x100 <= n && n < NUM_TAGS {
            w.write_all(&[0xf0 | (n >> 8) as u8, n as u8])
        } else {
            Err(old_io::IoError {
                kind: old_io::OtherIoError,
                desc: "invalid tag",
                detail: Some(format!("{}", n))
            })
        }
    }

    fn write_sized_vuint<W: Writer>(w: &mut W, n: uint, size: uint) -> EncodeResult {
        match size {
            1 => w.write_all(&[0x80u8 | (n as u8)]),
            2 => w.write_all(&[0x40u8 | ((n >> 8) as u8), n as u8]),
            3 => w.write_all(&[0x20u8 | ((n >> 16) as u8), (n >> 8) as u8,
                            n as u8]),
            4 => w.write_all(&[0x10u8 | ((n >> 24) as u8), (n >> 16) as u8,
                            (n >> 8) as u8, n as u8]),
            _ => Err(old_io::IoError {
                kind: old_io::OtherIoError,
                desc: "int too big",
                detail: Some(format!("{}", n))
            })
        }
    }

    fn write_vuint<W: Writer>(w: &mut W, n: uint) -> EncodeResult {
        if n < 0x7f { return write_sized_vuint(w, n, 1); }
        if n < 0x4000 { return write_sized_vuint(w, n, 2); }
        if n < 0x200000 { return write_sized_vuint(w, n, 3); }
        if n < 0x10000000 { return write_sized_vuint(w, n, 4); }
        Err(old_io::IoError {
            kind: old_io::OtherIoError,
            desc: "int too big",
            detail: Some(format!("{}", n))
        })
    }

    impl<'a, W: Writer + Seek> Encoder<'a, W> {
        pub fn new(w: &'a mut W) -> Encoder<'a, W> {
            Encoder {
                writer: w,
                size_positions: vec!(),
            }
        }

        /// FIXME(pcwalton): Workaround for badness in trans. DO NOT USE ME.
        pub unsafe fn unsafe_clone(&self) -> Encoder<'a, W> {
            Encoder {
                writer: mem::transmute_copy(&self.writer),
                size_positions: self.size_positions.clone(),
            }
        }

        pub fn start_tag(&mut self, tag_id: uint) -> EncodeResult {
            debug!("Start tag {:?}", tag_id);
            assert!(tag_id >= NUM_IMPLICIT_TAGS);

            // Write the enum ID:
            try!(write_tag(self.writer, tag_id));

            // Write a placeholder four-byte size.
            self.size_positions.push(try!(self.writer.tell()) as uint);
            let zeroes: &[u8] = &[0u8, 0u8, 0u8, 0u8];
            self.writer.write_all(zeroes)
        }

        pub fn end_tag(&mut self) -> EncodeResult {
            let last_size_pos = self.size_positions.pop().unwrap();
            let cur_pos = try!(self.writer.tell());
            try!(self.writer.seek(last_size_pos as i64, old_io::SeekSet));
            let size = cur_pos as uint - last_size_pos - 4;
            try!(write_sized_vuint(self.writer, size, 4));
            let r = try!(self.writer.seek(cur_pos as i64, old_io::SeekSet));

            debug!("End tag (size = {:?})", size);
            Ok(r)
        }

        pub fn wr_tag<F>(&mut self, tag_id: uint, blk: F) -> EncodeResult where
            F: FnOnce() -> EncodeResult,
        {
            try!(self.start_tag(tag_id));
            try!(blk());
            self.end_tag()
        }

        pub fn wr_tagged_bytes(&mut self, tag_id: uint, b: &[u8]) -> EncodeResult {
            assert!(tag_id >= NUM_IMPLICIT_TAGS);
            try!(write_tag(self.writer, tag_id));
            try!(write_vuint(self.writer, b.len()));
            self.writer.write_all(b)
        }

        pub fn wr_tagged_u64(&mut self, tag_id: uint, v: u64) -> EncodeResult {
            let bytes: [u8; 8] = unsafe { mem::transmute(v.to_be()) };
            self.wr_tagged_bytes(tag_id, &bytes)
        }

        pub fn wr_tagged_u32(&mut self, tag_id: uint, v: u32)  -> EncodeResult{
            let bytes: [u8; 4] = unsafe { mem::transmute(v.to_be()) };
            self.wr_tagged_bytes(tag_id, &bytes)
        }

        pub fn wr_tagged_u16(&mut self, tag_id: uint, v: u16) -> EncodeResult {
            let bytes: [u8; 2] = unsafe { mem::transmute(v.to_be()) };
            self.wr_tagged_bytes(tag_id, &bytes)
        }

        pub fn wr_tagged_u8(&mut self, tag_id: uint, v: u8) -> EncodeResult {
            self.wr_tagged_bytes(tag_id, &[v])
        }

        pub fn wr_tagged_i64(&mut self, tag_id: uint, v: i64) -> EncodeResult {
            self.wr_tagged_u64(tag_id, v as u64)
        }

        pub fn wr_tagged_i32(&mut self, tag_id: uint, v: i32) -> EncodeResult {
            self.wr_tagged_u32(tag_id, v as u32)
        }

        pub fn wr_tagged_i16(&mut self, tag_id: uint, v: i16) -> EncodeResult {
            self.wr_tagged_u16(tag_id, v as u16)
        }

        pub fn wr_tagged_i8(&mut self, tag_id: uint, v: i8) -> EncodeResult {
            self.wr_tagged_bytes(tag_id, &[v as u8])
        }

        pub fn wr_tagged_str(&mut self, tag_id: uint, v: &str) -> EncodeResult {
            self.wr_tagged_bytes(tag_id, v.as_bytes())
        }

        // for auto-serialization
        fn wr_tagged_raw_bytes(&mut self, tag_id: uint, b: &[u8]) -> EncodeResult {
            try!(write_tag(self.writer, tag_id));
            self.writer.write_all(b)
        }

        fn wr_tagged_raw_u64(&mut self, tag_id: uint, v: u64) -> EncodeResult {
            let bytes: [u8; 8] = unsafe { mem::transmute(v.to_be()) };
            self.wr_tagged_raw_bytes(tag_id, &bytes)
        }

        fn wr_tagged_raw_u32(&mut self, tag_id: uint, v: u32)  -> EncodeResult{
            let bytes: [u8; 4] = unsafe { mem::transmute(v.to_be()) };
            self.wr_tagged_raw_bytes(tag_id, &bytes)
        }

        fn wr_tagged_raw_u16(&mut self, tag_id: uint, v: u16) -> EncodeResult {
            let bytes: [u8; 2] = unsafe { mem::transmute(v.to_be()) };
            self.wr_tagged_raw_bytes(tag_id, &bytes)
        }

        fn wr_tagged_raw_u8(&mut self, tag_id: uint, v: u8) -> EncodeResult {
            self.wr_tagged_raw_bytes(tag_id, &[v])
        }

        fn wr_tagged_raw_i64(&mut self, tag_id: uint, v: i64) -> EncodeResult {
            self.wr_tagged_raw_u64(tag_id, v as u64)
        }

        fn wr_tagged_raw_i32(&mut self, tag_id: uint, v: i32) -> EncodeResult {
            self.wr_tagged_raw_u32(tag_id, v as u32)
        }

        fn wr_tagged_raw_i16(&mut self, tag_id: uint, v: i16) -> EncodeResult {
            self.wr_tagged_raw_u16(tag_id, v as u16)
        }

        fn wr_tagged_raw_i8(&mut self, tag_id: uint, v: i8) -> EncodeResult {
            self.wr_tagged_raw_bytes(tag_id, &[v as u8])
        }

        pub fn wr_bytes(&mut self, b: &[u8]) -> EncodeResult {
            debug!("Write {:?} bytes", b.len());
            self.writer.write_all(b)
        }

        pub fn wr_str(&mut self, s: &str) -> EncodeResult {
            debug!("Write str: {:?}", s);
            self.writer.write_all(s.as_bytes())
        }
    }

    // FIXME (#2743): optionally perform "relaxations" on end_tag to more
    // efficiently encode sizes; this is a fixed point iteration

    impl<'a, W: Writer + Seek> Encoder<'a, W> {
        // used internally to emit things like the vector length and so on
        fn _emit_tagged_sub(&mut self, v: uint) -> EncodeResult {
            if let Some(v) = v.to_u8() {
                self.wr_tagged_raw_u8(EsSub8 as uint, v)
            } else if let Some(v) = v.to_u32() {
                self.wr_tagged_raw_u32(EsSub32 as uint, v)
            } else {
                Err(old_io::IoError {
                    kind: old_io::OtherIoError,
                    desc: "length or variant id too big",
                    detail: Some(format!("{}", v))
                })
            }
        }

        pub fn emit_opaque<F>(&mut self, f: F) -> EncodeResult where
            F: FnOnce(&mut Encoder<W>) -> EncodeResult,
        {
            try!(self.start_tag(EsOpaque as uint));
            try!(f(self));
            self.end_tag()
        }
    }

    impl<'a, W: Writer + Seek> serialize::Encoder for Encoder<'a, W> {
        type Error = old_io::IoError;

        fn emit_nil(&mut self) -> EncodeResult {
            Ok(())
        }

        fn emit_uint(&mut self, v: uint) -> EncodeResult {
            self.wr_tagged_raw_u64(EsUint as uint, v as u64)
        }
        fn emit_u64(&mut self, v: u64) -> EncodeResult {
            self.wr_tagged_raw_u64(EsU64 as uint, v)
        }
        fn emit_u32(&mut self, v: u32) -> EncodeResult {
            self.wr_tagged_raw_u32(EsU32 as uint, v)
        }
        fn emit_u16(&mut self, v: u16) -> EncodeResult {
            self.wr_tagged_raw_u16(EsU16 as uint, v)
        }
        fn emit_u8(&mut self, v: u8) -> EncodeResult {
            self.wr_tagged_raw_u8(EsU8 as uint, v)
        }

        fn emit_int(&mut self, v: int) -> EncodeResult {
            self.wr_tagged_raw_i64(EsInt as uint, v as i64)
        }
        fn emit_i64(&mut self, v: i64) -> EncodeResult {
            self.wr_tagged_raw_i64(EsI64 as uint, v)
        }
        fn emit_i32(&mut self, v: i32) -> EncodeResult {
            self.wr_tagged_raw_i32(EsI32 as uint, v)
        }
        fn emit_i16(&mut self, v: i16) -> EncodeResult {
            self.wr_tagged_raw_i16(EsI16 as uint, v)
        }
        fn emit_i8(&mut self, v: i8) -> EncodeResult {
            self.wr_tagged_raw_i8(EsI8 as uint, v)
        }

        fn emit_bool(&mut self, v: bool) -> EncodeResult {
            self.wr_tagged_raw_u8(EsBool as uint, v as u8)
        }

        fn emit_f64(&mut self, v: f64) -> EncodeResult {
            let bits = unsafe { mem::transmute(v) };
            self.wr_tagged_raw_u64(EsF64 as uint, bits)
        }
        fn emit_f32(&mut self, v: f32) -> EncodeResult {
            let bits = unsafe { mem::transmute(v) };
            self.wr_tagged_raw_u32(EsF32 as uint, bits)
        }
        fn emit_char(&mut self, v: char) -> EncodeResult {
            self.wr_tagged_raw_u32(EsChar as uint, v as u32)
        }

        fn emit_str(&mut self, v: &str) -> EncodeResult {
            self.wr_tagged_str(EsStr as uint, v)
        }

        fn emit_enum<F>(&mut self, _name: &str, f: F) -> EncodeResult where
            F: FnOnce(&mut Encoder<'a, W>) -> EncodeResult,
        {
            try!(self.start_tag(EsEnum as uint));
            try!(f(self));
            self.end_tag()
        }

        fn emit_enum_variant<F>(&mut self,
                                _: &str,
                                v_id: uint,
                                _: uint,
                                f: F) -> EncodeResult where
            F: FnOnce(&mut Encoder<'a, W>) -> EncodeResult,
        {
            try!(self._emit_tagged_sub(v_id));
            f(self)
        }

        fn emit_enum_variant_arg<F>(&mut self, _: uint, f: F) -> EncodeResult where
            F: FnOnce(&mut Encoder<'a, W>) -> EncodeResult,
        {
            f(self)
        }

        fn emit_enum_struct_variant<F>(&mut self,
                                       v_name: &str,
                                       v_id: uint,
                                       cnt: uint,
                                       f: F) -> EncodeResult where
            F: FnOnce(&mut Encoder<'a, W>) -> EncodeResult,
        {
            self.emit_enum_variant(v_name, v_id, cnt, f)
        }

        fn emit_enum_struct_variant_field<F>(&mut self,
                                             _: &str,
                                             idx: uint,
                                             f: F) -> EncodeResult where
            F: FnOnce(&mut Encoder<'a, W>) -> EncodeResult,
        {
            self.emit_enum_variant_arg(idx, f)
        }

        fn emit_struct<F>(&mut self, _: &str, _len: uint, f: F) -> EncodeResult where
            F: FnOnce(&mut Encoder<'a, W>) -> EncodeResult,
        {
            f(self)
        }

        fn emit_struct_field<F>(&mut self, _name: &str, _: uint, f: F) -> EncodeResult where
            F: FnOnce(&mut Encoder<'a, W>) -> EncodeResult,
        {
            f(self)
        }

        fn emit_tuple<F>(&mut self, len: uint, f: F) -> EncodeResult where
            F: FnOnce(&mut Encoder<'a, W>) -> EncodeResult,
        {
            self.emit_seq(len, f)
        }
        fn emit_tuple_arg<F>(&mut self, idx: uint, f: F) -> EncodeResult where
            F: FnOnce(&mut Encoder<'a, W>) -> EncodeResult,
        {
            self.emit_seq_elt(idx, f)
        }

        fn emit_tuple_struct<F>(&mut self, _: &str, len: uint, f: F) -> EncodeResult where
            F: FnOnce(&mut Encoder<'a, W>) -> EncodeResult,
        {
            self.emit_seq(len, f)
        }
        fn emit_tuple_struct_arg<F>(&mut self, idx: uint, f: F) -> EncodeResult where
            F: FnOnce(&mut Encoder<'a, W>) -> EncodeResult,
        {
            self.emit_seq_elt(idx, f)
        }

        fn emit_option<F>(&mut self, f: F) -> EncodeResult where
            F: FnOnce(&mut Encoder<'a, W>) -> EncodeResult,
        {
            self.emit_enum("Option", f)
        }
        fn emit_option_none(&mut self) -> EncodeResult {
            self.emit_enum_variant("None", 0, 0, |_| Ok(()))
        }
        fn emit_option_some<F>(&mut self, f: F) -> EncodeResult where
            F: FnOnce(&mut Encoder<'a, W>) -> EncodeResult,
        {

            self.emit_enum_variant("Some", 1, 1, f)
        }

        fn emit_seq<F>(&mut self, len: uint, f: F) -> EncodeResult where
            F: FnOnce(&mut Encoder<'a, W>) -> EncodeResult,
        {

            try!(self.start_tag(EsVec as uint));
            try!(self._emit_tagged_sub(len));
            try!(f(self));
            self.end_tag()
        }

        fn emit_seq_elt<F>(&mut self, _idx: uint, f: F) -> EncodeResult where
            F: FnOnce(&mut Encoder<'a, W>) -> EncodeResult,
        {

            try!(self.start_tag(EsVecElt as uint));
            try!(f(self));
            self.end_tag()
        }

        fn emit_map<F>(&mut self, len: uint, f: F) -> EncodeResult where
            F: FnOnce(&mut Encoder<'a, W>) -> EncodeResult,
        {

            try!(self.start_tag(EsMap as uint));
            try!(self._emit_tagged_sub(len));
            try!(f(self));
            self.end_tag()
        }

        fn emit_map_elt_key<F>(&mut self, _idx: uint, f: F) -> EncodeResult where
            F: FnOnce(&mut Encoder<'a, W>) -> EncodeResult,
        {

            try!(self.start_tag(EsMapKey as uint));
            try!(f(self));
            self.end_tag()
        }

        fn emit_map_elt_val<F>(&mut self, _idx: uint, f: F) -> EncodeResult where
            F: FnOnce(&mut Encoder<'a, W>) -> EncodeResult,
        {
            try!(self.start_tag(EsMapVal as uint));
            try!(f(self));
            self.end_tag()
        }
    }
}

// ___________________________________________________________________________
// Testing

#[cfg(test)]
mod tests {
    use super::{Doc, reader, writer};
    use super::io::SeekableMemWriter;

    use serialize::{Encodable, Decodable};

    use std::option::Option;
    use std::option::Option::{None, Some};

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

        let mut res: reader::Res;

        // Class A
        res = reader::vuint_at(data, 0).unwrap();
        assert_eq!(res.val, 0);
        assert_eq!(res.next, 1);
        res = reader::vuint_at(data, res.next).unwrap();
        assert_eq!(res.val, (1 << 7) - 1);
        assert_eq!(res.next, 2);

        // Class B
        res = reader::vuint_at(data, res.next).unwrap();
        assert_eq!(res.val, 0);
        assert_eq!(res.next, 4);
        res = reader::vuint_at(data, res.next).unwrap();
        assert_eq!(res.val, (1 << 14) - 1);
        assert_eq!(res.next, 6);

        // Class C
        res = reader::vuint_at(data, res.next).unwrap();
        assert_eq!(res.val, 0);
        assert_eq!(res.next, 9);
        res = reader::vuint_at(data, res.next).unwrap();
        assert_eq!(res.val, (1 << 21) - 1);
        assert_eq!(res.next, 12);

        // Class D
        res = reader::vuint_at(data, res.next).unwrap();
        assert_eq!(res.val, 0);
        assert_eq!(res.next, 16);
        res = reader::vuint_at(data, res.next).unwrap();
        assert_eq!(res.val, (1 << 28) - 1);
        assert_eq!(res.next, 20);
    }

    #[test]
    fn test_option_int() {
        fn test_v(v: Option<int>) {
            debug!("v == {:?}", v);
            let mut wr = SeekableMemWriter::new();
            {
                let mut rbml_w = writer::Encoder::new(&mut wr);
                let _ = v.encode(&mut rbml_w);
            }
            let rbml_doc = Doc::new(wr.get_ref());
            let mut deser = reader::Decoder::new(rbml_doc);
            let v1 = Decodable::decode(&mut deser).unwrap();
            debug!("v1 == {:?}", v1);
            assert_eq!(v, v1);
        }

        test_v(Some(22));
        test_v(None);
        test_v(Some(3));
    }
}

#[cfg(test)]
mod bench {
    #![allow(non_snake_case)]
    use test::Bencher;
    use super::reader;

    #[bench]
    pub fn vuint_at_A_aligned(b: &mut Bencher) {
        let data = (0i32..4*100).map(|i| {
            match i % 2 {
              0 => 0x80u8,
              _ => i as u8,
            }
        }).collect::<Vec<_>>();
        let mut sum = 0;
        b.iter(|| {
            let mut i = 0;
            while i < data.len() {
                sum += reader::vuint_at(&data, i).unwrap().val;
                i += 4;
            }
        });
    }

    #[bench]
    pub fn vuint_at_A_unaligned(b: &mut Bencher) {
        let data = (0i32..4*100+1).map(|i| {
            match i % 2 {
              1 => 0x80u8,
              _ => i as u8
            }
        }).collect::<Vec<_>>();
        let mut sum = 0;
        b.iter(|| {
            let mut i = 1;
            while i < data.len() {
                sum += reader::vuint_at(&data, i).unwrap().val;
                i += 4;
            }
        });
    }

    #[bench]
    pub fn vuint_at_D_aligned(b: &mut Bencher) {
        let data = (0i32..4*100).map(|i| {
            match i % 4 {
              0 => 0x10u8,
              3 => i as u8,
              _ => 0u8
            }
        }).collect::<Vec<_>>();
        let mut sum = 0;
        b.iter(|| {
            let mut i = 0;
            while i < data.len() {
                sum += reader::vuint_at(&data, i).unwrap().val;
                i += 4;
            }
        });
    }

    #[bench]
    pub fn vuint_at_D_unaligned(b: &mut Bencher) {
        let data = (0i32..4*100+1).map(|i| {
            match i % 4 {
              1 => 0x10u8,
              0 => i as u8,
              _ => 0u8
            }
        }).collect::<Vec<_>>();
        let mut sum = 0;
        b.iter(|| {
            let mut i = 1;
            while i < data.len() {
                sum += reader::vuint_at(&data, i).unwrap().val;
                i += 4;
            }
        });
    }
}
