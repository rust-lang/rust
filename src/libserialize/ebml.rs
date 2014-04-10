// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(missing_doc)]

use std::io;
use std::str;

// Simple Extensible Binary Markup Language (ebml) reader and writer on a
// cursor model. See the specification here:
//     http://www.matroska.org/technical/specs/rfc/index.html

// Common data structures
#[deriving(Clone)]
pub struct Doc<'a> {
    pub data: &'a [u8],
    pub start: uint,
    pub end: uint,
}

impl<'doc> Doc<'doc> {
    pub fn get<'a>(&'a self, tag: uint) -> Doc<'a> {
        reader::get_doc(*self, tag)
    }

    pub fn as_str_slice<'a>(&'a self) -> &'a str {
        str::from_utf8(self.data.slice(self.start, self.end)).unwrap()
    }

    pub fn as_str(&self) -> ~str {
        self.as_str_slice().to_owned()
    }
}

pub struct TaggedDoc<'a> {
    tag: uint,
    pub doc: Doc<'a>,
}

pub enum EbmlEncoderTag {
    EsUint,     // 0
    EsU64,      // 1
    EsU32,      // 2
    EsU16,      // 3
    EsU8,       // 4
    EsInt,      // 5
    EsI64,      // 6
    EsI32,      // 7
    EsI16,      // 8
    EsI8,       // 9
    EsBool,     // 10
    EsChar,     // 11
    EsStr,      // 12
    EsF64,      // 13
    EsF32,      // 14
    EsFloat,    // 15
    EsEnum,     // 16
    EsEnumVid,  // 17
    EsEnumBody, // 18
    EsVec,      // 19
    EsVecLen,   // 20
    EsVecElt,   // 21
    EsMap,      // 22
    EsMapLen,   // 23
    EsMapKey,   // 24
    EsMapVal,   // 25

    EsOpaque,

    EsLabel, // Used only when debugging
}

#[deriving(Show)]
pub enum Error {
    IntTooBig(uint),
    Expected(~str),
    IoError(io::IoError)
}
// --------------------------------------

pub mod reader {
    use std::char;

    use std::cast::transmute;
    use std::int;
    use std::option::{None, Option, Some};
    use std::io::extensions::u64_from_be_bytes;

    use serialize;

    use super::{ EsVec, EsMap, EsEnum, EsVecLen, EsVecElt, EsMapLen, EsMapKey,
        EsEnumVid, EsU64, EsU32, EsU16, EsU8, EsInt, EsI64, EsI32, EsI16, EsI8,
        EsBool, EsF64, EsF32, EsChar, EsStr, EsMapVal, EsEnumBody, EsUint,
        EsOpaque, EsLabel, EbmlEncoderTag, Doc, TaggedDoc, Error, IntTooBig,
        Expected };

    pub type DecodeResult<T> = Result<T, Error>;
    // ebml reading

    macro_rules! try_or(
        ($e:expr, $r:expr) => (
            match $e {
                Ok(e) => e,
                Err(e) => {
                    debug!("ignored error: {}", e);
                    return $r
                }
            }
        )
    )

    pub struct Res {
        pub val: uint,
        pub next: uint
    }

    #[inline(never)]
    fn vuint_at_slow(data: &[u8], start: uint) -> DecodeResult<Res> {
        let a = data[start];
        if a & 0x80u8 != 0u8 {
            return Ok(Res {val: (a & 0x7fu8) as uint, next: start + 1u});
        }
        if a & 0x40u8 != 0u8 {
            return Ok(Res {val: ((a & 0x3fu8) as uint) << 8u |
                        (data[start + 1u] as uint),
                    next: start + 2u});
        }
        if a & 0x20u8 != 0u8 {
            return Ok(Res {val: ((a & 0x1fu8) as uint) << 16u |
                        (data[start + 1u] as uint) << 8u |
                        (data[start + 2u] as uint),
                    next: start + 3u});
        }
        if a & 0x10u8 != 0u8 {
            return Ok(Res {val: ((a & 0x0fu8) as uint) << 24u |
                        (data[start + 1u] as uint) << 16u |
                        (data[start + 2u] as uint) << 8u |
                        (data[start + 3u] as uint),
                    next: start + 4u});
        }
        Err(IntTooBig(a as uint))
    }

    pub fn vuint_at(data: &[u8], start: uint) -> DecodeResult<Res> {
        use std::mem::from_be32;

        if data.len() - start < 4 {
            return vuint_at_slow(data, start);
        }

        // Lookup table for parsing EBML Element IDs as per http://ebml.sourceforge.net/specs/
        // The Element IDs are parsed by reading a big endian u32 positioned at data[start].
        // Using the four most significant bits of the u32 we lookup in the table below how the
        // element ID should be derived from it.
        //
        // The table stores tuples (shift, mask) where shift is the number the u32 should be right
        // shifted with and mask is the value the right shifted value should be masked with.
        // If for example the most significant bit is set this means it's a class A ID and the u32
        // should be right shifted with 24 and masked with 0x7f. Therefore we store (24, 0x7f) at
        // index 0x8 - 0xF (four bit numbers where the most significant bit is set).
        //
        // By storing the number of shifts and masks in a table instead of checking in order if
        // the most significant bit is set, the second most significant bit is set etc. we can
        // replace up to three "and+branch" with a single table lookup which gives us a measured
        // speedup of around 2x on x86_64.
        static SHIFT_MASK_TABLE: [(u32, u32), ..16] = [
            (0, 0x0), (0, 0x0fffffff),
            (8, 0x1fffff), (8, 0x1fffff),
            (16, 0x3fff), (16, 0x3fff), (16, 0x3fff), (16, 0x3fff),
            (24, 0x7f), (24, 0x7f), (24, 0x7f), (24, 0x7f),
            (24, 0x7f), (24, 0x7f), (24, 0x7f), (24, 0x7f)
        ];

        unsafe {
            let ptr = data.as_ptr().offset(start as int) as *i32;
            let val = from_be32(*ptr) as u32;

            let i = (val >> 28u) as uint;
            let (shift, mask) = SHIFT_MASK_TABLE[i];
            Ok(Res {
                val: ((val >> shift) & mask) as uint,
                next: start + (((32 - shift) >> 3) as uint)
            })
        }
    }

    pub fn Doc<'a>(data: &'a [u8]) -> Doc<'a> {
        Doc { data: data, start: 0u, end: data.len() }
    }

    pub fn doc_at<'a>(data: &'a [u8], start: uint) -> DecodeResult<TaggedDoc<'a>> {
        let elt_tag = try!(vuint_at(data, start));
        let elt_size = try!(vuint_at(data, elt_tag.next));
        let end = elt_size.next + elt_size.val;
        Ok(TaggedDoc {
            tag: elt_tag.val,
            doc: Doc { data: data, start: elt_size.next, end: end }
        })
    }

    pub fn maybe_get_doc<'a>(d: Doc<'a>, tg: uint) -> Option<Doc<'a>> {
        let mut pos = d.start;
        while pos < d.end {
            let elt_tag = try_or!(vuint_at(d.data, pos), None);
            let elt_size = try_or!(vuint_at(d.data, elt_tag.next), None);
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
                error!("failed to find block with tag {}", tg);
                fail!();
            }
        }
    }

    pub fn docs<'a>(d: Doc<'a>, it: |uint, Doc<'a>| -> bool) -> bool {
        let mut pos = d.start;
        while pos < d.end {
            let elt_tag = try_or!(vuint_at(d.data, pos), false);
            let elt_size = try_or!(vuint_at(d.data, elt_tag.next), false);
            pos = elt_size.next + elt_size.val;
            let doc = Doc { data: d.data, start: elt_size.next, end: pos };
            if !it(elt_tag.val, doc) {
                return false;
            }
        }
        return true;
    }

    pub fn tagged_docs<'a>(d: Doc<'a>, tg: uint, it: |Doc<'a>| -> bool) -> bool {
        let mut pos = d.start;
        while pos < d.end {
            let elt_tag = try_or!(vuint_at(d.data, pos), false);
            let elt_size = try_or!(vuint_at(d.data, elt_tag.next), false);
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

    pub fn with_doc_data<'a, T>(d: Doc<'a>, f: |x: &'a [u8]| -> T) -> T {
        f(d.data.slice(d.start, d.end))
    }


    pub fn doc_as_u8(d: Doc) -> u8 {
        assert_eq!(d.end, d.start + 1u);
        d.data[d.start]
    }

    pub fn doc_as_u16(d: Doc) -> u16 {
        assert_eq!(d.end, d.start + 2u);
        u64_from_be_bytes(d.data, d.start, 2u) as u16
    }

    pub fn doc_as_u32(d: Doc) -> u32 {
        assert_eq!(d.end, d.start + 4u);
        u64_from_be_bytes(d.data, d.start, 4u) as u32
    }

    pub fn doc_as_u64(d: Doc) -> u64 {
        assert_eq!(d.end, d.start + 8u);
        u64_from_be_bytes(d.data, d.start, 8u)
    }

    pub fn doc_as_i8(d: Doc) -> i8 { doc_as_u8(d) as i8 }
    pub fn doc_as_i16(d: Doc) -> i16 { doc_as_u16(d) as i16 }
    pub fn doc_as_i32(d: Doc) -> i32 { doc_as_u32(d) as i32 }
    pub fn doc_as_i64(d: Doc) -> i64 { doc_as_u64(d) as i64 }

    pub struct Decoder<'a> {
        parent: Doc<'a>,
        pos: uint,
    }

    pub fn Decoder<'a>(d: Doc<'a>) -> Decoder<'a> {
        Decoder {
            parent: d,
            pos: d.start
        }
    }

    impl<'doc> Decoder<'doc> {
        fn _check_label(&mut self, lbl: &str) -> DecodeResult<()> {
            if self.pos < self.parent.end {
                let TaggedDoc { tag: r_tag, doc: r_doc } =
                    try!(doc_at(self.parent.data, self.pos));

                if r_tag == (EsLabel as uint) {
                    self.pos = r_doc.end;
                    let str = r_doc.as_str_slice();
                    if lbl != str {
                        return Err(Expected(format!("Expected label {} but found {}", lbl, str)));
                    }
                }
            }
            Ok(())
        }

        fn next_doc(&mut self, exp_tag: EbmlEncoderTag) -> DecodeResult<Doc<'doc>> {
            debug!(". next_doc(exp_tag={:?})", exp_tag);
            if self.pos >= self.parent.end {
                return Err(Expected(format!("no more documents in current node!")));
            }
            let TaggedDoc { tag: r_tag, doc: r_doc } =
                try!(doc_at(self.parent.data, self.pos));
            debug!("self.parent={}-{} self.pos={} r_tag={} r_doc={}-{}",
                   self.parent.start,
                   self.parent.end,
                   self.pos,
                   r_tag,
                   r_doc.start,
                   r_doc.end);
            if r_tag != (exp_tag as uint) {
                return Err(Expected(format!("expected EBML doc with tag {:?} but found tag {:?}",
                       exp_tag, r_tag)));
            }
            if r_doc.end > self.parent.end {
                return Err(Expected(format!("invalid EBML, child extends to {:#x}, parent to {:#x}",
                      r_doc.end, self.parent.end)));
            }
            self.pos = r_doc.end;
            Ok(r_doc)
        }

        fn push_doc<T>(&mut self, exp_tag: EbmlEncoderTag,
                       f: |&mut Decoder<'doc>| -> DecodeResult<T>) -> DecodeResult<T> {
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

        fn _next_uint(&mut self, exp_tag: EbmlEncoderTag) -> DecodeResult<uint> {
            let r = doc_as_u32(try!(self.next_doc(exp_tag)));
            debug!("_next_uint exp_tag={:?} result={}", exp_tag, r);
            Ok(r as uint)
        }

        pub fn read_opaque<R>(&mut self,
                              op: |&mut Decoder<'doc>, Doc| -> DecodeResult<R>) -> DecodeResult<R> {
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

    impl<'doc> serialize::Decoder<Error> for Decoder<'doc> {
        fn read_nil(&mut self) -> DecodeResult<()> { Ok(()) }

        fn read_u64(&mut self) -> DecodeResult<u64> { Ok(doc_as_u64(try!(self.next_doc(EsU64)))) }
        fn read_u32(&mut self) -> DecodeResult<u32> { Ok(doc_as_u32(try!(self.next_doc(EsU32)))) }
        fn read_u16(&mut self) -> DecodeResult<u16> { Ok(doc_as_u16(try!(self.next_doc(EsU16)))) }
        fn read_u8 (&mut self) -> DecodeResult<u8 > { Ok(doc_as_u8 (try!(self.next_doc(EsU8 )))) }
        fn read_uint(&mut self) -> DecodeResult<uint> {
            let v = doc_as_u64(try!(self.next_doc(EsUint)));
            if v > (::std::uint::MAX as u64) {
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
            if v > (int::MAX as i64) || v < (int::MIN as i64) {
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
        fn read_str(&mut self) -> DecodeResult<~str> {
            Ok(try!(self.next_doc(EsStr)).as_str())
        }

        // Compound types:
        fn read_enum<T>(&mut self,
                        name: &str,
                        f: |&mut Decoder<'doc>| -> DecodeResult<T>) -> DecodeResult<T> {
            debug!("read_enum({})", name);
            try!(self._check_label(name));

            let doc = try!(self.next_doc(EsEnum));

            let (old_parent, old_pos) = (self.parent, self.pos);
            self.parent = doc;
            self.pos = self.parent.start;

            let result = try!(f(self));

            self.parent = old_parent;
            self.pos = old_pos;
            Ok(result)
        }

        fn read_enum_variant<T>(&mut self,
                                _: &[&str],
                                f: |&mut Decoder<'doc>, uint| -> DecodeResult<T>)
                                -> DecodeResult<T> {
            debug!("read_enum_variant()");
            let idx = try!(self._next_uint(EsEnumVid));
            debug!("  idx={}", idx);

            let doc = try!(self.next_doc(EsEnumBody));

            let (old_parent, old_pos) = (self.parent, self.pos);
            self.parent = doc;
            self.pos = self.parent.start;

            let result = try!(f(self, idx));

            self.parent = old_parent;
            self.pos = old_pos;
            Ok(result)
        }

        fn read_enum_variant_arg<T>(&mut self,
                                    idx: uint,
                                    f: |&mut Decoder<'doc>| -> DecodeResult<T>) -> DecodeResult<T> {
            debug!("read_enum_variant_arg(idx={})", idx);
            f(self)
        }

        fn read_enum_struct_variant<T>(&mut self,
                                       _: &[&str],
                                       f: |&mut Decoder<'doc>, uint| -> DecodeResult<T>)
                                       -> DecodeResult<T> {
            debug!("read_enum_struct_variant()");
            let idx = try!(self._next_uint(EsEnumVid));
            debug!("  idx={}", idx);

            let doc = try!(self.next_doc(EsEnumBody));

            let (old_parent, old_pos) = (self.parent, self.pos);
            self.parent = doc;
            self.pos = self.parent.start;

            let result = try!(f(self, idx));

            self.parent = old_parent;
            self.pos = old_pos;
            Ok(result)
        }

        fn read_enum_struct_variant_field<T>(&mut self,
                                             name: &str,
                                             idx: uint,
                                             f: |&mut Decoder<'doc>| -> DecodeResult<T>)
                                             -> DecodeResult<T> {
            debug!("read_enum_struct_variant_arg(name={}, idx={})", name, idx);
            f(self)
        }

        fn read_struct<T>(&mut self,
                          name: &str,
                          _: uint,
                          f: |&mut Decoder<'doc>| -> DecodeResult<T>)
                          -> DecodeResult<T> {
            debug!("read_struct(name={})", name);
            f(self)
        }

        fn read_struct_field<T>(&mut self,
                                name: &str,
                                idx: uint,
                                f: |&mut Decoder<'doc>| -> DecodeResult<T>)
                                -> DecodeResult<T> {
            debug!("read_struct_field(name={}, idx={})", name, idx);
            try!(self._check_label(name));
            f(self)
        }

        fn read_tuple<T>(&mut self,
                         f: |&mut Decoder<'doc>, uint| -> DecodeResult<T>) -> DecodeResult<T> {
            debug!("read_tuple()");
            self.read_seq(f)
        }

        fn read_tuple_arg<T>(&mut self, idx: uint, f: |&mut Decoder<'doc>| -> DecodeResult<T>)
                             -> DecodeResult<T> {
            debug!("read_tuple_arg(idx={})", idx);
            self.read_seq_elt(idx, f)
        }

        fn read_tuple_struct<T>(&mut self,
                                name: &str,
                                f: |&mut Decoder<'doc>, uint| -> DecodeResult<T>)
                                -> DecodeResult<T> {
            debug!("read_tuple_struct(name={})", name);
            self.read_tuple(f)
        }

        fn read_tuple_struct_arg<T>(&mut self,
                                    idx: uint,
                                    f: |&mut Decoder<'doc>| -> DecodeResult<T>)
                                    -> DecodeResult<T> {
            debug!("read_tuple_struct_arg(idx={})", idx);
            self.read_tuple_arg(idx, f)
        }

        fn read_option<T>(&mut self,
                          f: |&mut Decoder<'doc>, bool| -> DecodeResult<T>) -> DecodeResult<T> {
            debug!("read_option()");
            self.read_enum("Option", |this| {
                this.read_enum_variant(["None", "Some"], |this, idx| {
                    match idx {
                        0 => f(this, false),
                        1 => f(this, true),
                        _ => Err(Expected(format!("Expected None or Some"))),
                    }
                })
            })
        }

        fn read_seq<T>(&mut self,
                       f: |&mut Decoder<'doc>, uint| -> DecodeResult<T>) -> DecodeResult<T> {
            debug!("read_seq()");
            self.push_doc(EsVec, |d| {
                let len = try!(d._next_uint(EsVecLen));
                debug!("  len={}", len);
                f(d, len)
            })
        }

        fn read_seq_elt<T>(&mut self, idx: uint, f: |&mut Decoder<'doc>| -> DecodeResult<T>)
                           -> DecodeResult<T> {
            debug!("read_seq_elt(idx={})", idx);
            self.push_doc(EsVecElt, f)
        }

        fn read_map<T>(&mut self,
                       f: |&mut Decoder<'doc>, uint| -> DecodeResult<T>) -> DecodeResult<T> {
            debug!("read_map()");
            self.push_doc(EsMap, |d| {
                let len = try!(d._next_uint(EsMapLen));
                debug!("  len={}", len);
                f(d, len)
            })
        }

        fn read_map_elt_key<T>(&mut self, idx: uint, f: |&mut Decoder<'doc>| -> DecodeResult<T>)
                               -> DecodeResult<T> {
            debug!("read_map_elt_key(idx={})", idx);
            self.push_doc(EsMapKey, f)
        }

        fn read_map_elt_val<T>(&mut self, idx: uint, f: |&mut Decoder<'doc>| -> DecodeResult<T>)
                               -> DecodeResult<T> {
            debug!("read_map_elt_val(idx={})", idx);
            self.push_doc(EsMapVal, f)
        }
    }
}

pub mod writer {
    use std::cast;
    use std::clone::Clone;
    use std::io;
    use std::io::{Writer, Seek};
    use std::io::extensions::u64_to_be_bytes;

    use super::{ EsVec, EsMap, EsEnum, EsVecLen, EsVecElt, EsMapLen, EsMapKey,
        EsEnumVid, EsU64, EsU32, EsU16, EsU8, EsInt, EsI64, EsI32, EsI16, EsI8,
        EsBool, EsF64, EsF32, EsChar, EsStr, EsMapVal, EsEnumBody, EsUint,
        EsOpaque, EsLabel, EbmlEncoderTag };

    use serialize;


    pub type EncodeResult = io::IoResult<()>;

    // ebml writing
    pub struct Encoder<'a, W> {
        pub writer: &'a mut W,
        size_positions: Vec<uint>,
    }

    fn write_sized_vuint<W: Writer>(w: &mut W, n: uint, size: uint) -> EncodeResult {
        match size {
            1u => w.write(&[0x80u8 | (n as u8)]),
            2u => w.write(&[0x40u8 | ((n >> 8_u) as u8), n as u8]),
            3u => w.write(&[0x20u8 | ((n >> 16_u) as u8), (n >> 8_u) as u8,
                            n as u8]),
            4u => w.write(&[0x10u8 | ((n >> 24_u) as u8), (n >> 16_u) as u8,
                            (n >> 8_u) as u8, n as u8]),
            _ => Err(io::IoError {
                kind: io::OtherIoError,
                desc: "int too big",
                detail: Some(format!("{}", n))
            })
        }
    }

    fn write_vuint<W: Writer>(w: &mut W, n: uint) -> EncodeResult {
        if n < 0x7f_u { return write_sized_vuint(w, n, 1u); }
        if n < 0x4000_u { return write_sized_vuint(w, n, 2u); }
        if n < 0x200000_u { return write_sized_vuint(w, n, 3u); }
        if n < 0x10000000_u { return write_sized_vuint(w, n, 4u); }
        Err(io::IoError {
            kind: io::OtherIoError,
            desc: "int too big",
            detail: Some(format!("{}", n))
        })
    }

    pub fn Encoder<'a, W: Writer + Seek>(w: &'a mut W) -> Encoder<'a, W> {
        Encoder {
            writer: w,
            size_positions: vec!(),
        }
    }

    // FIXME (#2741): Provide a function to write the standard ebml header.
    impl<'a, W: Writer + Seek> Encoder<'a, W> {
        /// FIXME(pcwalton): Workaround for badness in trans. DO NOT USE ME.
        pub unsafe fn unsafe_clone(&self) -> Encoder<'a, W> {
            Encoder {
                writer: cast::transmute_copy(&self.writer),
                size_positions: self.size_positions.clone(),
            }
        }

        pub fn start_tag(&mut self, tag_id: uint) -> EncodeResult {
            debug!("Start tag {}", tag_id);

            // Write the enum ID:
            try!(write_vuint(self.writer, tag_id));

            // Write a placeholder four-byte size.
            self.size_positions.push(try!(self.writer.tell()) as uint);
            let zeroes: &[u8] = &[0u8, 0u8, 0u8, 0u8];
            self.writer.write(zeroes)
        }

        pub fn end_tag(&mut self) -> EncodeResult {
            let last_size_pos = self.size_positions.pop().unwrap();
            let cur_pos = try!(self.writer.tell());
            try!(self.writer.seek(last_size_pos as i64, io::SeekSet));
            let size = cur_pos as uint - last_size_pos - 4;
            try!(write_sized_vuint(self.writer, size, 4u));
            let r = try!(self.writer.seek(cur_pos as i64, io::SeekSet));

            debug!("End tag (size = {})", size);
            Ok(r)
        }

        pub fn wr_tag(&mut self, tag_id: uint, blk: || -> EncodeResult) -> EncodeResult {
            try!(self.start_tag(tag_id));
            try!(blk());
            self.end_tag()
        }

        pub fn wr_tagged_bytes(&mut self, tag_id: uint, b: &[u8]) -> EncodeResult {
            try!(write_vuint(self.writer, tag_id));
            try!(write_vuint(self.writer, b.len()));
            self.writer.write(b)
        }

        pub fn wr_tagged_u64(&mut self, tag_id: uint, v: u64) -> EncodeResult {
            u64_to_be_bytes(v, 8u, |v| {
                self.wr_tagged_bytes(tag_id, v)
            })
        }

        pub fn wr_tagged_u32(&mut self, tag_id: uint, v: u32)  -> EncodeResult{
            u64_to_be_bytes(v as u64, 4u, |v| {
                self.wr_tagged_bytes(tag_id, v)
            })
        }

        pub fn wr_tagged_u16(&mut self, tag_id: uint, v: u16) -> EncodeResult {
            u64_to_be_bytes(v as u64, 2u, |v| {
                self.wr_tagged_bytes(tag_id, v)
            })
        }

        pub fn wr_tagged_u8(&mut self, tag_id: uint, v: u8) -> EncodeResult {
            self.wr_tagged_bytes(tag_id, &[v])
        }

        pub fn wr_tagged_i64(&mut self, tag_id: uint, v: i64) -> EncodeResult {
            u64_to_be_bytes(v as u64, 8u, |v| {
                self.wr_tagged_bytes(tag_id, v)
            })
        }

        pub fn wr_tagged_i32(&mut self, tag_id: uint, v: i32) -> EncodeResult {
            u64_to_be_bytes(v as u64, 4u, |v| {
                self.wr_tagged_bytes(tag_id, v)
            })
        }

        pub fn wr_tagged_i16(&mut self, tag_id: uint, v: i16) -> EncodeResult {
            u64_to_be_bytes(v as u64, 2u, |v| {
                self.wr_tagged_bytes(tag_id, v)
            })
        }

        pub fn wr_tagged_i8(&mut self, tag_id: uint, v: i8) -> EncodeResult {
            self.wr_tagged_bytes(tag_id, &[v as u8])
        }

        pub fn wr_tagged_str(&mut self, tag_id: uint, v: &str) -> EncodeResult {
            self.wr_tagged_bytes(tag_id, v.as_bytes())
        }

        pub fn wr_bytes(&mut self, b: &[u8]) -> EncodeResult {
            debug!("Write {} bytes", b.len());
            self.writer.write(b)
        }

        pub fn wr_str(&mut self, s: &str) -> EncodeResult {
            debug!("Write str: {}", s);
            self.writer.write(s.as_bytes())
        }
    }

    // FIXME (#2743): optionally perform "relaxations" on end_tag to more
    // efficiently encode sizes; this is a fixed point iteration

    // Set to true to generate more debugging in EBML code.
    // Totally lame approach.
    static DEBUG: bool = true;

    impl<'a, W: Writer + Seek> Encoder<'a, W> {
        // used internally to emit things like the vector length and so on
        fn _emit_tagged_uint(&mut self, t: EbmlEncoderTag, v: uint) -> EncodeResult {
            assert!(v <= 0xFFFF_FFFF_u);
            self.wr_tagged_u32(t as uint, v as u32)
        }

        fn _emit_label(&mut self, label: &str) -> EncodeResult {
            // There are various strings that we have access to, such as
            // the name of a record field, which do not actually appear in
            // the encoded EBML (normally).  This is just for
            // efficiency.  When debugging, though, we can emit such
            // labels and then they will be checked by decoder to
            // try and check failures more quickly.
            if DEBUG { self.wr_tagged_str(EsLabel as uint, label) }
            else { Ok(()) }
        }

        pub fn emit_opaque(&mut self, f: |&mut Encoder<W>| -> EncodeResult) -> EncodeResult {
            try!(self.start_tag(EsOpaque as uint));
            try!(f(self));
            self.end_tag()
        }
    }

    impl<'a, W: Writer + Seek> serialize::Encoder<io::IoError> for Encoder<'a, W> {
        fn emit_nil(&mut self) -> EncodeResult {
            Ok(())
        }

        fn emit_uint(&mut self, v: uint) -> EncodeResult {
            self.wr_tagged_u64(EsUint as uint, v as u64)
        }
        fn emit_u64(&mut self, v: u64) -> EncodeResult {
            self.wr_tagged_u64(EsU64 as uint, v)
        }
        fn emit_u32(&mut self, v: u32) -> EncodeResult {
            self.wr_tagged_u32(EsU32 as uint, v)
        }
        fn emit_u16(&mut self, v: u16) -> EncodeResult {
            self.wr_tagged_u16(EsU16 as uint, v)
        }
        fn emit_u8(&mut self, v: u8) -> EncodeResult {
            self.wr_tagged_u8(EsU8 as uint, v)
        }

        fn emit_int(&mut self, v: int) -> EncodeResult {
            self.wr_tagged_i64(EsInt as uint, v as i64)
        }
        fn emit_i64(&mut self, v: i64) -> EncodeResult {
            self.wr_tagged_i64(EsI64 as uint, v)
        }
        fn emit_i32(&mut self, v: i32) -> EncodeResult {
            self.wr_tagged_i32(EsI32 as uint, v)
        }
        fn emit_i16(&mut self, v: i16) -> EncodeResult {
            self.wr_tagged_i16(EsI16 as uint, v)
        }
        fn emit_i8(&mut self, v: i8) -> EncodeResult {
            self.wr_tagged_i8(EsI8 as uint, v)
        }

        fn emit_bool(&mut self, v: bool) -> EncodeResult {
            self.wr_tagged_u8(EsBool as uint, v as u8)
        }

        fn emit_f64(&mut self, v: f64) -> EncodeResult {
            let bits = unsafe { cast::transmute(v) };
            self.wr_tagged_u64(EsF64 as uint, bits)
        }
        fn emit_f32(&mut self, v: f32) -> EncodeResult {
            let bits = unsafe { cast::transmute(v) };
            self.wr_tagged_u32(EsF32 as uint, bits)
        }
        fn emit_char(&mut self, v: char) -> EncodeResult {
            self.wr_tagged_u32(EsChar as uint, v as u32)
        }

        fn emit_str(&mut self, v: &str) -> EncodeResult {
            self.wr_tagged_str(EsStr as uint, v)
        }

        fn emit_enum(&mut self,
                     name: &str,
                     f: |&mut Encoder<'a, W>| -> EncodeResult) -> EncodeResult {
            try!(self._emit_label(name));
            try!(self.start_tag(EsEnum as uint));
            try!(f(self));
            self.end_tag()
        }

        fn emit_enum_variant(&mut self,
                             _: &str,
                             v_id: uint,
                             _: uint,
                             f: |&mut Encoder<'a, W>| -> EncodeResult) -> EncodeResult {
            try!(self._emit_tagged_uint(EsEnumVid, v_id));
            try!(self.start_tag(EsEnumBody as uint));
            try!(f(self));
            self.end_tag()
        }

        fn emit_enum_variant_arg(&mut self,
                                 _: uint,
                                 f: |&mut Encoder<'a, W>| -> EncodeResult) -> EncodeResult {
            f(self)
        }

        fn emit_enum_struct_variant(&mut self,
                                    v_name: &str,
                                    v_id: uint,
                                    cnt: uint,
                                    f: |&mut Encoder<'a, W>| -> EncodeResult) -> EncodeResult {
            self.emit_enum_variant(v_name, v_id, cnt, f)
        }

        fn emit_enum_struct_variant_field(&mut self,
                                          _: &str,
                                          idx: uint,
                                          f: |&mut Encoder<'a, W>| -> EncodeResult)
            -> EncodeResult {
            self.emit_enum_variant_arg(idx, f)
        }

        fn emit_struct(&mut self,
                       _: &str,
                       _len: uint,
                       f: |&mut Encoder<'a, W>| -> EncodeResult) -> EncodeResult {
            f(self)
        }

        fn emit_struct_field(&mut self,
                             name: &str,
                             _: uint,
                             f: |&mut Encoder<'a, W>| -> EncodeResult) -> EncodeResult {
            try!(self._emit_label(name));
            f(self)
        }

        fn emit_tuple(&mut self,
                      len: uint,
                      f: |&mut Encoder<'a, W>| -> EncodeResult) -> EncodeResult {
            self.emit_seq(len, f)
        }
        fn emit_tuple_arg(&mut self,
                          idx: uint,
                          f: |&mut Encoder<'a, W>| -> EncodeResult) -> EncodeResult {
            self.emit_seq_elt(idx, f)
        }

        fn emit_tuple_struct(&mut self,
                             _: &str,
                             len: uint,
                             f: |&mut Encoder<'a, W>| -> EncodeResult) -> EncodeResult {
            self.emit_seq(len, f)
        }
        fn emit_tuple_struct_arg(&mut self,
                                 idx: uint,
                                 f: |&mut Encoder<'a, W>| -> EncodeResult) -> EncodeResult {
            self.emit_seq_elt(idx, f)
        }

        fn emit_option(&mut self,
                       f: |&mut Encoder<'a, W>| -> EncodeResult) -> EncodeResult {
            self.emit_enum("Option", f)
        }
        fn emit_option_none(&mut self) -> EncodeResult {
            self.emit_enum_variant("None", 0, 0, |_| Ok(()))
        }
        fn emit_option_some(&mut self,
                            f: |&mut Encoder<'a, W>| -> EncodeResult) -> EncodeResult {

            self.emit_enum_variant("Some", 1, 1, f)
        }

        fn emit_seq(&mut self,
                    len: uint,
                    f: |&mut Encoder<'a, W>| -> EncodeResult) -> EncodeResult {

            try!(self.start_tag(EsVec as uint));
            try!(self._emit_tagged_uint(EsVecLen, len));
            try!(f(self));
            self.end_tag()
        }

        fn emit_seq_elt(&mut self,
                        _idx: uint,
                        f: |&mut Encoder<'a, W>| -> EncodeResult) -> EncodeResult {

            try!(self.start_tag(EsVecElt as uint));
            try!(f(self));
            self.end_tag()
        }

        fn emit_map(&mut self,
                    len: uint,
                    f: |&mut Encoder<'a, W>| -> EncodeResult) -> EncodeResult {

            try!(self.start_tag(EsMap as uint));
            try!(self._emit_tagged_uint(EsMapLen, len));
            try!(f(self));
            self.end_tag()
        }

        fn emit_map_elt_key(&mut self,
                            _idx: uint,
                            f: |&mut Encoder<'a, W>| -> EncodeResult) -> EncodeResult {

            try!(self.start_tag(EsMapKey as uint));
            try!(f(self));
            self.end_tag()
        }

        fn emit_map_elt_val(&mut self,
                            _idx: uint,
                            f: |&mut Encoder<'a, W>| -> EncodeResult) -> EncodeResult {
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
    use ebml::reader;
    use ebml::writer;
    use {Encodable, Decodable};

    use std::io::MemWriter;
    use std::option::{None, Option, Some};

    #[test]
    fn test_vuint_at() {
        let data = [
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
            let mut wr = MemWriter::new();
            {
                let mut ebml_w = writer::Encoder(&mut wr);
                let _ = v.encode(&mut ebml_w);
            }
            let ebml_doc = reader::Doc(wr.get_ref());
            let mut deser = reader::Decoder(ebml_doc);
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
    extern crate test;
    use self::test::BenchHarness;
    use ebml::reader;

    #[bench]
    pub fn vuint_at_A_aligned(bh: &mut BenchHarness) {
        use std::slice;
        let data = slice::from_fn(4*100, |i| {
            match i % 2 {
              0 => 0x80u8,
              _ => i as u8,
            }
        });
        let mut sum = 0u;
        bh.iter(|| {
            let mut i = 0;
            while i < data.len() {
                sum += reader::vuint_at(data, i).unwrap().val;
                i += 4;
            }
        });
    }

    #[bench]
    pub fn vuint_at_A_unaligned(bh: &mut BenchHarness) {
        use std::slice;
        let data = slice::from_fn(4*100+1, |i| {
            match i % 2 {
              1 => 0x80u8,
              _ => i as u8
            }
        });
        let mut sum = 0u;
        bh.iter(|| {
            let mut i = 1;
            while i < data.len() {
                sum += reader::vuint_at(data, i).unwrap().val;
                i += 4;
            }
        });
    }

    #[bench]
    pub fn vuint_at_D_aligned(bh: &mut BenchHarness) {
        use std::slice;
        let data = slice::from_fn(4*100, |i| {
            match i % 4 {
              0 => 0x10u8,
              3 => i as u8,
              _ => 0u8
            }
        });
        let mut sum = 0u;
        bh.iter(|| {
            let mut i = 0;
            while i < data.len() {
                sum += reader::vuint_at(data, i).unwrap().val;
                i += 4;
            }
        });
    }

    #[bench]
    pub fn vuint_at_D_unaligned(bh: &mut BenchHarness) {
        use std::slice;
        let data = slice::from_fn(4*100+1, |i| {
            match i % 4 {
              1 => 0x10u8,
              0 => i as u8,
              _ => 0u8
            }
        });
        let mut sum = 0u;
        bh.iter(|| {
            let mut i = 1;
            while i < data.len() {
                sum += reader::vuint_at(data, i).unwrap().val;
                i += 4;
            }
        });
    }
}
