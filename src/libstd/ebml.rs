// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

// Simple Extensible Binary Markup Language (ebml) reader and writer on a
// cursor model. See the specification here:
//     http://www.matroska.org/technical/specs/rfc/index.html

// Common data structures
struct EbmlTag {
    id: uint,
    size: uint,
}

struct EbmlState {
    ebml_tag: EbmlTag,
    tag_pos: uint,
    data_pos: uint,
}

pub struct Doc {
    data: @~[u8],
    start: uint,
    end: uint,
}

pub struct TaggedDoc {
    tag: uint,
    doc: Doc,
}

pub enum EbmlEncoderTag {
    EsUint, EsU64, EsU32, EsU16, EsU8,
    EsInt, EsI64, EsI32, EsI16, EsI8,
    EsBool,
    EsStr,
    EsF64, EsF32, EsFloat,
    EsEnum, EsEnumVid, EsEnumBody,
    EsVec, EsVecLen, EsVecElt,

    EsOpaque,

    EsLabel // Used only when debugging
}
// --------------------------------------

pub mod reader {
    use ebml::{Doc, EbmlEncoderTag, EsBool, EsEnum, EsEnumBody, EsEnumVid};
    use ebml::{EsI16, EsI32, EsI64, EsI8, EsInt};
    use ebml::{EsLabel, EsOpaque, EsStr, EsU16, EsU32, EsU64, EsU8, EsUint};
    use ebml::{EsVec, EsVecElt, EsVecLen, TaggedDoc};
    use serialize;

    use core::int;
    use core::io;
    use core::prelude::*;
    use core::str;
    use core::vec;

    // ebml reading

    pub impl Doc {
        fn get(&self, tag: uint) -> Doc {
            get_doc(*self, tag)
        }
    }

    struct Res {
        val: uint,
        next: uint
    }

    fn vuint_at(data: &[u8], start: uint) -> Res {
        let a = data[start];
        if a & 0x80u8 != 0u8 {
            return Res {val: (a & 0x7fu8) as uint, next: start + 1u};
        }
        if a & 0x40u8 != 0u8 {
            return Res {val: ((a & 0x3fu8) as uint) << 8u |
                        (data[start + 1u] as uint),
                    next: start + 2u};
        } else if a & 0x20u8 != 0u8 {
            return Res {val: ((a & 0x1fu8) as uint) << 16u |
                        (data[start + 1u] as uint) << 8u |
                        (data[start + 2u] as uint),
                    next: start + 3u};
        } else if a & 0x10u8 != 0u8 {
            return Res {val: ((a & 0x0fu8) as uint) << 24u |
                        (data[start + 1u] as uint) << 16u |
                        (data[start + 2u] as uint) << 8u |
                        (data[start + 3u] as uint),
                    next: start + 4u};
        } else { error!("vint too big"); fail!(); }
    }

    pub fn Doc(data: @~[u8]) -> Doc {
        Doc { data: data, start: 0u, end: vec::len::<u8>(*data) }
    }

    pub fn doc_at(data: @~[u8], start: uint) -> TaggedDoc {
        let elt_tag = vuint_at(*data, start);
        let elt_size = vuint_at(*data, elt_tag.next);
        let end = elt_size.next + elt_size.val;
        TaggedDoc {
            tag: elt_tag.val,
            doc: Doc { data: data, start: elt_size.next, end: end }
        }
    }

    pub fn maybe_get_doc(d: Doc, tg: uint) -> Option<Doc> {
        let mut pos = d.start;
        while pos < d.end {
            let elt_tag = vuint_at(*d.data, pos);
            let elt_size = vuint_at(*d.data, elt_tag.next);
            pos = elt_size.next + elt_size.val;
            if elt_tag.val == tg {
                return Some(Doc { data: d.data, start: elt_size.next,
                                  end: pos });
            }
        }
        None
    }

    pub fn get_doc(d: Doc, tg: uint) -> Doc {
        match maybe_get_doc(d, tg) {
            Some(d) => d,
            None => {
                error!("failed to find block with tag %u", tg);
                fail!();
            }
        }
    }

    pub fn docs(d: Doc, it: &fn(uint, Doc) -> bool) {
        let mut pos = d.start;
        while pos < d.end {
            let elt_tag = vuint_at(*d.data, pos);
            let elt_size = vuint_at(*d.data, elt_tag.next);
            pos = elt_size.next + elt_size.val;
            let doc = Doc { data: d.data, start: elt_size.next, end: pos };
            if !it(elt_tag.val, doc) {
                break;
            }
        }
    }

    pub fn tagged_docs(d: Doc, tg: uint, it: &fn(Doc) -> bool) {
        let mut pos = d.start;
        while pos < d.end {
            let elt_tag = vuint_at(*d.data, pos);
            let elt_size = vuint_at(*d.data, elt_tag.next);
            pos = elt_size.next + elt_size.val;
            if elt_tag.val == tg {
                let doc = Doc { data: d.data, start: elt_size.next,
                                end: pos };
                if !it(doc) {
                    break;
                }
            }
        }
    }

    pub fn doc_data(d: Doc) -> ~[u8] {
        vec::slice::<u8>(*d.data, d.start, d.end).to_vec()
    }

    pub fn with_doc_data<T>(d: Doc, f: &fn(x: &[u8]) -> T) -> T {
        f(vec::slice(*d.data, d.start, d.end))
    }

    pub fn doc_as_str(d: Doc) -> ~str { str::from_bytes(doc_data(d)) }

    pub fn doc_as_u8(d: Doc) -> u8 {
        assert!(d.end == d.start + 1u);
        (*d.data)[d.start]
    }

    pub fn doc_as_u16(d: Doc) -> u16 {
        assert!(d.end == d.start + 2u);
        io::u64_from_be_bytes(*d.data, d.start, 2u) as u16
    }

    pub fn doc_as_u32(d: Doc) -> u32 {
        assert!(d.end == d.start + 4u);
        io::u64_from_be_bytes(*d.data, d.start, 4u) as u32
    }

    pub fn doc_as_u64(d: Doc) -> u64 {
        assert!(d.end == d.start + 8u);
        io::u64_from_be_bytes(*d.data, d.start, 8u)
    }

    pub fn doc_as_i8(d: Doc) -> i8 { doc_as_u8(d) as i8 }
    pub fn doc_as_i16(d: Doc) -> i16 { doc_as_u16(d) as i16 }
    pub fn doc_as_i32(d: Doc) -> i32 { doc_as_u32(d) as i32 }
    pub fn doc_as_i64(d: Doc) -> i64 { doc_as_u64(d) as i64 }


    pub struct Decoder {
        priv mut parent: Doc,
        priv mut pos: uint,
    }

    pub fn Decoder(d: Doc) -> Decoder {
        Decoder { parent: d, pos: d.start }
    }

    priv impl Decoder {
        fn _check_label(&self, lbl: &str) {
            if self.pos < self.parent.end {
                let TaggedDoc { tag: r_tag, doc: r_doc } =
                    doc_at(self.parent.data, self.pos);

                if r_tag == (EsLabel as uint) {
                    self.pos = r_doc.end;
                    let str = doc_as_str(r_doc);
                    if lbl != str {
                        fail!(fmt!("Expected label %s but found %s", lbl,
                            str));
                    }
                }
            }
        }

        fn next_doc(&self, exp_tag: EbmlEncoderTag) -> Doc {
            debug!(". next_doc(exp_tag=%?)", exp_tag);
            if self.pos >= self.parent.end {
                fail!(~"no more documents in current node!");
            }
            let TaggedDoc { tag: r_tag, doc: r_doc } =
                doc_at(self.parent.data, self.pos);
            debug!("self.parent=%?-%? self.pos=%? r_tag=%? r_doc=%?-%?",
                   copy self.parent.start, copy self.parent.end,
                   copy self.pos, r_tag, r_doc.start, r_doc.end);
            if r_tag != (exp_tag as uint) {
                fail!(fmt!("expected EBML doc with tag %? but found tag %?",
                          exp_tag, r_tag));
            }
            if r_doc.end > self.parent.end {
                fail!(fmt!("invalid EBML, child extends to 0x%x, \
                           parent to 0x%x", r_doc.end, self.parent.end));
            }
            self.pos = r_doc.end;
            r_doc
        }

        fn push_doc<T>(&self, d: Doc, f: &fn() -> T) -> T {
            let old_parent = self.parent;
            let old_pos = self.pos;
            self.parent = d;
            self.pos = d.start;
            let r = f();
            self.parent = old_parent;
            self.pos = old_pos;
            r
        }

        fn _next_uint(&self, exp_tag: EbmlEncoderTag) -> uint {
            let r = doc_as_u32(self.next_doc(exp_tag));
            debug!("_next_uint exp_tag=%? result=%?", exp_tag, r);
            r as uint
        }
    }

    pub impl Decoder {
        fn read_opaque<R>(&self, op: &fn(Doc) -> R) -> R {
            do self.push_doc(self.next_doc(EsOpaque)) {
                op(copy self.parent)
            }
        }
    }

    impl serialize::Decoder for Decoder {
        fn read_nil(&self) -> () { () }

        fn read_u64(&self) -> u64 { doc_as_u64(self.next_doc(EsU64)) }
        fn read_u32(&self) -> u32 { doc_as_u32(self.next_doc(EsU32)) }
        fn read_u16(&self) -> u16 { doc_as_u16(self.next_doc(EsU16)) }
        fn read_u8 (&self) -> u8  { doc_as_u8 (self.next_doc(EsU8 )) }
        fn read_uint(&self) -> uint {
            let v = doc_as_u64(self.next_doc(EsUint));
            if v > (::core::uint::max_value as u64) {
                fail!(fmt!("uint %? too large for this architecture", v));
            }
            v as uint
        }

        fn read_i64(&self) -> i64 { doc_as_u64(self.next_doc(EsI64)) as i64 }
        fn read_i32(&self) -> i32 { doc_as_u32(self.next_doc(EsI32)) as i32 }
        fn read_i16(&self) -> i16 { doc_as_u16(self.next_doc(EsI16)) as i16 }
        fn read_i8 (&self) -> i8  { doc_as_u8 (self.next_doc(EsI8 )) as i8  }
        fn read_int(&self) -> int {
            let v = doc_as_u64(self.next_doc(EsInt)) as i64;
            if v > (int::max_value as i64) || v < (int::min_value as i64) {
                fail!(fmt!("int %? out of range for this architecture", v));
            }
            v as int
        }

        fn read_bool(&self) -> bool { doc_as_u8(self.next_doc(EsBool))
                                         as bool }

        fn read_f64(&self) -> f64 { fail!(~"read_f64()"); }
        fn read_f32(&self) -> f32 { fail!(~"read_f32()"); }
        fn read_float(&self) -> float { fail!(~"read_float()"); }

        fn read_char(&self) -> char { fail!(~"read_char()"); }

        fn read_owned_str(&self) -> ~str { doc_as_str(self.next_doc(EsStr)) }
        fn read_managed_str(&self) -> @str { fail!(~"read_managed_str()"); }

        // Compound types:
        fn read_owned<T>(&self, f: &fn() -> T) -> T {
            debug!("read_owned()");
            f()
        }

        fn read_managed<T>(&self, f: &fn() -> T) -> T {
            debug!("read_managed()");
            f()
        }

        fn read_enum<T>(&self, name: &str, f: &fn() -> T) -> T {
            debug!("read_enum(%s)", name);
            self._check_label(name);
            self.push_doc(self.next_doc(EsEnum), f)
        }

        fn read_enum_variant<T>(&self, _names: &[&str], f: &fn(uint) -> T) -> T {
            debug!("read_enum_variant()");
            let idx = self._next_uint(EsEnumVid);
            debug!("  idx=%u", idx);
            do self.push_doc(self.next_doc(EsEnumBody)) {
                f(idx)
            }
        }

        fn read_enum_variant_arg<T>(&self, idx: uint, f: &fn() -> T) -> T {
            debug!("read_enum_variant_arg(idx=%u)", idx);
            f()
        }

        fn read_owned_vec<T>(&self, f: &fn(uint) -> T) -> T {
            debug!("read_owned_vec()");
            do self.push_doc(self.next_doc(EsVec)) {
                let len = self._next_uint(EsVecLen);
                debug!("  len=%u", len);
                f(len)
            }
        }

        fn read_managed_vec<T>(&self, f: &fn(uint) -> T) -> T {
            debug!("read_managed_vec()");
            do self.push_doc(self.next_doc(EsVec)) {
                let len = self._next_uint(EsVecLen);
                debug!("  len=%u", len);
                f(len)
            }
        }

        fn read_vec_elt<T>(&self, idx: uint, f: &fn() -> T) -> T {
            debug!("read_vec_elt(idx=%u)", idx);
            self.push_doc(self.next_doc(EsVecElt), f)
        }

        fn read_rec<T>(&self, f: &fn() -> T) -> T {
            debug!("read_rec()");
            f()
        }

        fn read_struct<T>(&self, name: &str, _len: uint, f: &fn() -> T) -> T {
            debug!("read_struct(name=%s)", name);
            f()
        }

        fn read_field<T>(&self, name: &str, idx: uint, f: &fn() -> T) -> T {
            debug!("read_field(name=%s, idx=%u)", name, idx);
            self._check_label(name);
            f()
        }

        fn read_tup<T>(&self, len: uint, f: &fn() -> T) -> T {
            debug!("read_tup(len=%u)", len);
            f()
        }

        fn read_tup_elt<T>(&self, idx: uint, f: &fn() -> T) -> T {
            debug!("read_tup_elt(idx=%u)", idx);
            f()
        }

        fn read_option<T>(&self, f: &fn(bool) -> T) -> T {
            debug!("read_option()");
            do self.read_enum("Option") || {
                do self.read_enum_variant(["None", "Some"]) |idx| {
                    match idx {
                        0 => f(false),
                        1 => f(true),
                        _ => fail!(),
                    }
                }
            }
        }
    }
}

pub mod writer {
    use ebml::{EbmlEncoderTag, EsBool, EsEnum, EsEnumBody, EsEnumVid};
    use ebml::{EsI16, EsI32, EsI64, EsI8, EsInt};
    use ebml::{EsLabel, EsOpaque, EsStr, EsU16, EsU32, EsU64, EsU8, EsUint};
    use ebml::{EsVec, EsVecElt, EsVecLen};

    use core::io;
    use core::str;
    use core::vec;

    // ebml writing
    pub struct Encoder {
        writer: @io::Writer,
        priv mut size_positions: ~[uint],
    }

    fn write_sized_vuint(w: @io::Writer, n: uint, size: uint) {
        match size {
            1u => w.write(&[0x80u8 | (n as u8)]),
            2u => w.write(&[0x40u8 | ((n >> 8_u) as u8), n as u8]),
            3u => w.write(&[0x20u8 | ((n >> 16_u) as u8), (n >> 8_u) as u8,
                            n as u8]),
            4u => w.write(&[0x10u8 | ((n >> 24_u) as u8), (n >> 16_u) as u8,
                            (n >> 8_u) as u8, n as u8]),
            _ => fail!(fmt!("vint to write too big: %?", n))
        };
    }

    fn write_vuint(w: @io::Writer, n: uint) {
        if n < 0x7f_u { write_sized_vuint(w, n, 1u); return; }
        if n < 0x4000_u { write_sized_vuint(w, n, 2u); return; }
        if n < 0x200000_u { write_sized_vuint(w, n, 3u); return; }
        if n < 0x10000000_u { write_sized_vuint(w, n, 4u); return; }
        fail!(fmt!("vint to write too big: %?", n));
    }

    pub fn Encoder(w: @io::Writer) -> Encoder {
        let size_positions: ~[uint] = ~[];
        Encoder { writer: w, mut size_positions: size_positions }
    }

    // FIXME (#2741): Provide a function to write the standard ebml header.
    pub impl Encoder {
        fn start_tag(&self, tag_id: uint) {
            debug!("Start tag %u", tag_id);

            // Write the enum ID:
            write_vuint(self.writer, tag_id);

            // Write a placeholder four-byte size.
            self.size_positions.push(self.writer.tell());
            let zeroes: &[u8] = &[0u8, 0u8, 0u8, 0u8];
            self.writer.write(zeroes);
        }

        fn end_tag(&self) {
            let last_size_pos = self.size_positions.pop();
            let cur_pos = self.writer.tell();
            self.writer.seek(last_size_pos as int, io::SeekSet);
            let size = (cur_pos - last_size_pos - 4u);
            write_sized_vuint(self.writer, size, 4u);
            self.writer.seek(cur_pos as int, io::SeekSet);

            debug!("End tag (size = %u)", size);
        }

        fn wr_tag(&self, tag_id: uint, blk: &fn()) {
            self.start_tag(tag_id);
            blk();
            self.end_tag();
        }

        fn wr_tagged_bytes(&self, tag_id: uint, b: &[u8]) {
            write_vuint(self.writer, tag_id);
            write_vuint(self.writer, vec::len(b));
            self.writer.write(b);
        }

        fn wr_tagged_u64(&self, tag_id: uint, v: u64) {
            do io::u64_to_be_bytes(v, 8u) |v| {
                self.wr_tagged_bytes(tag_id, v);
            }
        }

        fn wr_tagged_u32(&self, tag_id: uint, v: u32) {
            do io::u64_to_be_bytes(v as u64, 4u) |v| {
                self.wr_tagged_bytes(tag_id, v);
            }
        }

        fn wr_tagged_u16(&self, tag_id: uint, v: u16) {
            do io::u64_to_be_bytes(v as u64, 2u) |v| {
                self.wr_tagged_bytes(tag_id, v);
            }
        }

        fn wr_tagged_u8(&self, tag_id: uint, v: u8) {
            self.wr_tagged_bytes(tag_id, &[v]);
        }

        fn wr_tagged_i64(&self, tag_id: uint, v: i64) {
            do io::u64_to_be_bytes(v as u64, 8u) |v| {
                self.wr_tagged_bytes(tag_id, v);
            }
        }

        fn wr_tagged_i32(&self, tag_id: uint, v: i32) {
            do io::u64_to_be_bytes(v as u64, 4u) |v| {
                self.wr_tagged_bytes(tag_id, v);
            }
        }

        fn wr_tagged_i16(&self, tag_id: uint, v: i16) {
            do io::u64_to_be_bytes(v as u64, 2u) |v| {
                self.wr_tagged_bytes(tag_id, v);
            }
        }

        fn wr_tagged_i8(&self, tag_id: uint, v: i8) {
            self.wr_tagged_bytes(tag_id, &[v as u8]);
        }

        fn wr_tagged_str(&self, tag_id: uint, v: &str) {
            str::byte_slice(v, |b| self.wr_tagged_bytes(tag_id, b));
        }

        fn wr_bytes(&self, b: &[u8]) {
            debug!("Write %u bytes", vec::len(b));
            self.writer.write(b);
        }

        fn wr_str(&self, s: &str) {
            debug!("Write str: %?", s);
            self.writer.write(str::to_bytes(s));
        }
    }

    // FIXME (#2743): optionally perform "relaxations" on end_tag to more
    // efficiently encode sizes; this is a fixed point iteration

    // Set to true to generate more debugging in EBML code.
    // Totally lame approach.
    static debug: bool = false;

    priv impl Encoder {
        // used internally to emit things like the vector length and so on
        fn _emit_tagged_uint(&self, t: EbmlEncoderTag, v: uint) {
            assert!(v <= 0xFFFF_FFFF_u);
            self.wr_tagged_u32(t as uint, v as u32);
        }

        fn _emit_label(&self, label: &str) {
            // There are various strings that we have access to, such as
            // the name of a record field, which do not actually appear in
            // the encoded EBML (normally).  This is just for
            // efficiency.  When debugging, though, we can emit such
            // labels and then they will be checked by decoder to
            // try and check failures more quickly.
            if debug { self.wr_tagged_str(EsLabel as uint, label) }
        }
    }

    pub impl Encoder {
        fn emit_opaque(&self, f: &fn()) {
            do self.wr_tag(EsOpaque as uint) {
                f()
            }
        }
    }

    impl ::serialize::Encoder for Encoder {
        fn emit_nil(&self) {}

        fn emit_uint(&self, v: uint) {
            self.wr_tagged_u64(EsUint as uint, v as u64);
        }
        fn emit_u64(&self, v: u64) { self.wr_tagged_u64(EsU64 as uint, v); }
        fn emit_u32(&self, v: u32) { self.wr_tagged_u32(EsU32 as uint, v); }
        fn emit_u16(&self, v: u16) { self.wr_tagged_u16(EsU16 as uint, v); }
        fn emit_u8(&self, v: u8)   { self.wr_tagged_u8 (EsU8  as uint, v); }

        fn emit_int(&self, v: int) {
            self.wr_tagged_i64(EsInt as uint, v as i64);
        }
        fn emit_i64(&self, v: i64) { self.wr_tagged_i64(EsI64 as uint, v); }
        fn emit_i32(&self, v: i32) { self.wr_tagged_i32(EsI32 as uint, v); }
        fn emit_i16(&self, v: i16) { self.wr_tagged_i16(EsI16 as uint, v); }
        fn emit_i8(&self, v: i8)   { self.wr_tagged_i8 (EsI8  as uint, v); }

        fn emit_bool(&self, v: bool) {
            self.wr_tagged_u8(EsBool as uint, v as u8)
        }

        // FIXME (#2742): implement these
        fn emit_f64(&self, _v: f64) {
            fail!(~"Unimplemented: serializing an f64");
        }
        fn emit_f32(&self, _v: f32) {
            fail!(~"Unimplemented: serializing an f32");
        }
        fn emit_float(&self, _v: float) {
            fail!(~"Unimplemented: serializing a float");
        }

        fn emit_char(&self, _v: char) {
            fail!(~"Unimplemented: serializing a char");
        }

        fn emit_borrowed_str(&self, v: &str) {
            self.wr_tagged_str(EsStr as uint, v)
        }

        fn emit_owned_str(&self, v: &str) {
            self.emit_borrowed_str(v)
        }

        fn emit_managed_str(&self, v: &str) {
            self.emit_borrowed_str(v)
        }

        fn emit_borrowed(&self, f: &fn()) { f() }
        fn emit_owned(&self, f: &fn()) { f() }
        fn emit_managed(&self, f: &fn()) { f() }

        fn emit_enum(&self, name: &str, f: &fn()) {
            self._emit_label(name);
            self.wr_tag(EsEnum as uint, f)
        }
        fn emit_enum_variant(&self, _v_name: &str, v_id: uint, _cnt: uint,
                             f: &fn()) {
            self._emit_tagged_uint(EsEnumVid, v_id);
            self.wr_tag(EsEnumBody as uint, f)
        }
        fn emit_enum_variant_arg(&self, _idx: uint, f: &fn()) { f() }

        fn emit_borrowed_vec(&self, len: uint, f: &fn()) {
            do self.wr_tag(EsVec as uint) {
                self._emit_tagged_uint(EsVecLen, len);
                f()
            }
        }

        fn emit_owned_vec(&self, len: uint, f: &fn()) {
            self.emit_borrowed_vec(len, f)
        }

        fn emit_managed_vec(&self, len: uint, f: &fn()) {
            self.emit_borrowed_vec(len, f)
        }

        fn emit_vec_elt(&self, _idx: uint, f: &fn()) {
            self.wr_tag(EsVecElt as uint, f)
        }

        fn emit_rec(&self, f: &fn()) { f() }
        fn emit_struct(&self, _name: &str, _len: uint, f: &fn()) { f() }
        fn emit_field(&self, name: &str, _idx: uint, f: &fn()) {
            self._emit_label(name);
            f()
        }

        fn emit_tup(&self, _len: uint, f: &fn()) { f() }
        fn emit_tup_elt(&self, _idx: uint, f: &fn()) { f() }

        fn emit_option(&self, f: &fn()) {
            self.emit_enum("Option", f);
        }
        fn emit_option_none(&self) {
            self.emit_enum_variant("None", 0, 0, || ())
        }
        fn emit_option_some(&self, f: &fn()) {
            self.emit_enum_variant("Some", 1, 1, f)
        }
    }
}

// ___________________________________________________________________________
// Testing

#[cfg(test)]
mod tests {
    use ebml::reader;
    use ebml::writer;
    use serialize::Encodable;
    use serialize;

    use core::io;
    use core::option::{None, Option, Some};

    #[test]
    fn test_option_int() {
        fn test_v(v: Option<int>) {
            debug!("v == %?", v);
            let bytes = do io::with_bytes_writer |wr| {
                let ebml_w = writer::Encoder(wr);
                v.encode(&ebml_w)
            };
            let ebml_doc = reader::Doc(@bytes);
            let deser = reader::Decoder(ebml_doc);
            let v1 = serialize::Decodable::decode(&deser);
            debug!("v1 == %?", v1);
            assert!(v == v1);
        }

        test_v(Some(22));
        test_v(None);
        test_v(Some(3));
    }
}
