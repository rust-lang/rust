// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(missing_doc)];

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
// --------------------------------------

pub mod reader {
    use super::*;

    use serialize;

    use core::prelude::*;
    use core::cast::transmute;
    use core::int;
    use core::io;
    use core::ptr::offset;
    use core::str;
    use core::unstable::intrinsics::bswap32;
    use core::vec;

    // ebml reading

    impl Doc {
        pub fn get(&self, tag: uint) -> Doc {
            get_doc(*self, tag)
        }
    }

    struct Res {
        val: uint,
        next: uint
    }

    #[inline(never)]
    fn vuint_at_slow(data: &[u8], start: uint) -> Res {
        let a = data[start];
        if a & 0x80u8 != 0u8 {
            return Res {val: (a & 0x7fu8) as uint, next: start + 1u};
        }
        if a & 0x40u8 != 0u8 {
            return Res {val: ((a & 0x3fu8) as uint) << 8u |
                        (data[start + 1u] as uint),
                    next: start + 2u};
        }
        if a & 0x20u8 != 0u8 {
            return Res {val: ((a & 0x1fu8) as uint) << 16u |
                        (data[start + 1u] as uint) << 8u |
                        (data[start + 2u] as uint),
                    next: start + 3u};
        }
        if a & 0x10u8 != 0u8 {
            return Res {val: ((a & 0x0fu8) as uint) << 24u |
                        (data[start + 1u] as uint) << 16u |
                        (data[start + 2u] as uint) << 8u |
                        (data[start + 3u] as uint),
                    next: start + 4u};
        }
        fail!("vint too big");
    }

    #[cfg(target_arch = "x86")]
    #[cfg(target_arch = "x86_64")]
    pub fn vuint_at(data: &[u8], start: uint) -> Res {
        if data.len() - start < 4 {
            return vuint_at_slow(data, start);
        }

        unsafe {
            let (ptr, _): (*u8, uint) = transmute(data);
            let ptr = offset(ptr, start);
            let ptr: *i32 = transmute(ptr);
            let val = bswap32(*ptr);
            let val: u32 = transmute(val);
            if (val & 0x80000000) != 0 {
                Res {
                    val: ((val >> 24) & 0x7f) as uint,
                    next: start + 1
                }
            } else if (val & 0x40000000) != 0 {
                Res {
                    val: ((val >> 16) & 0x3fff) as uint,
                    next: start + 2
                }
            } else if (val & 0x20000000) != 0 {
                Res {
                    val: ((val >> 8) & 0x1fffff) as uint,
                    next: start + 3
                }
            } else {
                Res {
                    val: (val & 0x0fffffff) as uint,
                    next: start + 4
                }
            }
        }
    }

    #[cfg(target_arch = "arm")]
    #[cfg(target_arch = "mips")]
    pub fn vuint_at(data: &[u8], start: uint) -> Res {
        vuint_at_slow(data, start)
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

    pub fn docs(d: Doc, it: &fn(uint, Doc) -> bool) -> bool {
        let mut pos = d.start;
        while pos < d.end {
            let elt_tag = vuint_at(*d.data, pos);
            let elt_size = vuint_at(*d.data, elt_tag.next);
            pos = elt_size.next + elt_size.val;
            let doc = Doc { data: d.data, start: elt_size.next, end: pos };
            if !it(elt_tag.val, doc) {
                return false;
            }
        }
        return true;
    }

    pub fn tagged_docs(d: Doc, tg: uint, it: &fn(Doc) -> bool) -> bool {
        let mut pos = d.start;
        while pos < d.end {
            let elt_tag = vuint_at(*d.data, pos);
            let elt_size = vuint_at(*d.data, elt_tag.next);
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

    pub fn doc_data(d: Doc) -> ~[u8] {
        vec::slice::<u8>(*d.data, d.start, d.end).to_vec()
    }

    pub fn with_doc_data<T>(d: Doc, f: &fn(x: &[u8]) -> T) -> T {
        f(vec::slice(*d.data, d.start, d.end))
    }

    pub fn doc_as_str(d: Doc) -> ~str { str::from_bytes(doc_data(d)) }

    pub fn doc_as_u8(d: Doc) -> u8 {
        assert_eq!(d.end, d.start + 1u);
        (*d.data)[d.start]
    }

    pub fn doc_as_u16(d: Doc) -> u16 {
        assert_eq!(d.end, d.start + 2u);
        io::u64_from_be_bytes(*d.data, d.start, 2u) as u16
    }

    pub fn doc_as_u32(d: Doc) -> u32 {
        assert_eq!(d.end, d.start + 4u);
        io::u64_from_be_bytes(*d.data, d.start, 4u) as u32
    }

    pub fn doc_as_u64(d: Doc) -> u64 {
        assert_eq!(d.end, d.start + 8u);
        io::u64_from_be_bytes(*d.data, d.start, 8u)
    }

    pub fn doc_as_i8(d: Doc) -> i8 { doc_as_u8(d) as i8 }
    pub fn doc_as_i16(d: Doc) -> i16 { doc_as_u16(d) as i16 }
    pub fn doc_as_i32(d: Doc) -> i32 { doc_as_u32(d) as i32 }
    pub fn doc_as_i64(d: Doc) -> i64 { doc_as_u64(d) as i64 }

    pub struct Decoder {
        priv parent: Doc,
        priv pos: uint,
    }

    pub fn Decoder(d: Doc) -> Decoder {
        Decoder {
            parent: d,
            pos: d.start
        }
    }

    impl Decoder {
        fn _check_label(&mut self, lbl: &str) {
            if self.pos < self.parent.end {
                let TaggedDoc { tag: r_tag, doc: r_doc } =
                    doc_at(self.parent.data, self.pos);

                if r_tag == (EsLabel as uint) {
                    self.pos = r_doc.end;
                    let str = doc_as_str(r_doc);
                    if lbl != str {
                        fail!("Expected label %s but found %s", lbl, str);
                    }
                }
            }
        }

        fn next_doc(&mut self, exp_tag: EbmlEncoderTag) -> Doc {
            debug!(". next_doc(exp_tag=%?)", exp_tag);
            if self.pos >= self.parent.end {
                fail!("no more documents in current node!");
            }
            let TaggedDoc { tag: r_tag, doc: r_doc } =
                doc_at(self.parent.data, self.pos);
            debug!("self.parent=%?-%? self.pos=%? r_tag=%? r_doc=%?-%?",
                   copy self.parent.start, copy self.parent.end,
                   copy self.pos, r_tag, r_doc.start, r_doc.end);
            if r_tag != (exp_tag as uint) {
                fail!("expected EBML doc with tag %? but found tag %?", exp_tag, r_tag);
            }
            if r_doc.end > self.parent.end {
                fail!("invalid EBML, child extends to 0x%x, parent to 0x%x",
                      r_doc.end, self.parent.end);
            }
            self.pos = r_doc.end;
            r_doc
        }

        fn push_doc<T>(&mut self, exp_tag: EbmlEncoderTag,
                       f: &fn(&mut Decoder) -> T) -> T {
            let d = self.next_doc(exp_tag);
            let old_parent = self.parent;
            let old_pos = self.pos;
            self.parent = d;
            self.pos = d.start;
            let r = f(self);
            self.parent = old_parent;
            self.pos = old_pos;
            r
        }

        fn _next_uint(&mut self, exp_tag: EbmlEncoderTag) -> uint {
            let r = doc_as_u32(self.next_doc(exp_tag));
            debug!("_next_uint exp_tag=%? result=%?", exp_tag, r);
            r as uint
        }
    }

    impl Decoder {
        pub fn read_opaque<R>(&mut self, op: &fn(&mut Decoder, Doc) -> R)
                              -> R {
            let doc = self.next_doc(EsOpaque);

            let (old_parent, old_pos) = (self.parent, self.pos);
            self.parent = doc;
            self.pos = doc.start;

            let result = op(self, doc);

            self.parent = old_parent;
            self.pos = old_pos;
            result
        }
    }

    impl serialize::Decoder for Decoder {
        fn read_nil(&mut self) -> () { () }

        fn read_u64(&mut self) -> u64 { doc_as_u64(self.next_doc(EsU64)) }
        fn read_u32(&mut self) -> u32 { doc_as_u32(self.next_doc(EsU32)) }
        fn read_u16(&mut self) -> u16 { doc_as_u16(self.next_doc(EsU16)) }
        fn read_u8 (&mut self) -> u8  { doc_as_u8 (self.next_doc(EsU8 )) }
        fn read_uint(&mut self) -> uint {
            let v = doc_as_u64(self.next_doc(EsUint));
            if v > (::core::uint::max_value as u64) {
                fail!("uint %? too large for this architecture", v);
            }
            v as uint
        }

        fn read_i64(&mut self) -> i64 {
            doc_as_u64(self.next_doc(EsI64)) as i64
        }
        fn read_i32(&mut self) -> i32 {
            doc_as_u32(self.next_doc(EsI32)) as i32
        }
        fn read_i16(&mut self) -> i16 {
            doc_as_u16(self.next_doc(EsI16)) as i16
        }
        fn read_i8 (&mut self) -> i8 {
            doc_as_u8(self.next_doc(EsI8 )) as i8
        }
        fn read_int(&mut self) -> int {
            let v = doc_as_u64(self.next_doc(EsInt)) as i64;
            if v > (int::max_value as i64) || v < (int::min_value as i64) {
                debug!("FIXME #6122: Removing this makes this function miscompile");
                fail!("int %? out of range for this architecture", v);
            }
            v as int
        }

        fn read_bool(&mut self) -> bool {
            doc_as_u8(self.next_doc(EsBool)) as bool
        }

        fn read_f64(&mut self) -> f64 {
            let bits = doc_as_u64(self.next_doc(EsF64));
            unsafe { transmute(bits) }
        }
        fn read_f32(&mut self) -> f32 {
            let bits = doc_as_u32(self.next_doc(EsF32));
            unsafe { transmute(bits) }
        }
        fn read_float(&mut self) -> float {
            let bits = doc_as_u64(self.next_doc(EsFloat));
            (unsafe { transmute::<u64, f64>(bits) }) as float
        }
        fn read_char(&mut self) -> char {
            doc_as_u32(self.next_doc(EsChar)) as char
        }
        fn read_str(&mut self) -> ~str { doc_as_str(self.next_doc(EsStr)) }

        // Compound types:
        fn read_enum<T>(&mut self,
                        name: &str,
                        f: &fn(&mut Decoder) -> T)
                        -> T {
            debug!("read_enum(%s)", name);
            self._check_label(name);

            let doc = self.next_doc(EsEnum);

            let (old_parent, old_pos) = (self.parent, self.pos);
            self.parent = doc;
            self.pos = self.parent.start;

            let result = f(self);

            self.parent = old_parent;
            self.pos = old_pos;
            result
        }

        fn read_enum_variant<T>(&mut self,
                                _: &[&str],
                                f: &fn(&mut Decoder, uint) -> T)
                                -> T {
            debug!("read_enum_variant()");
            let idx = self._next_uint(EsEnumVid);
            debug!("  idx=%u", idx);

            let doc = self.next_doc(EsEnumBody);

            let (old_parent, old_pos) = (self.parent, self.pos);
            self.parent = doc;
            self.pos = self.parent.start;

            let result = f(self, idx);

            self.parent = old_parent;
            self.pos = old_pos;
            result
        }

        fn read_enum_variant_arg<T>(&mut self,
                                    idx: uint,
                                    f: &fn(&mut Decoder) -> T) -> T {
            debug!("read_enum_variant_arg(idx=%u)", idx);
            f(self)
        }

        fn read_enum_struct_variant<T>(&mut self,
                                       _: &[&str],
                                       f: &fn(&mut Decoder, uint) -> T)
                                       -> T {
            debug!("read_enum_struct_variant()");
            let idx = self._next_uint(EsEnumVid);
            debug!("  idx=%u", idx);

            let doc = self.next_doc(EsEnumBody);

            let (old_parent, old_pos) = (self.parent, self.pos);
            self.parent = doc;
            self.pos = self.parent.start;

            let result = f(self, idx);

            self.parent = old_parent;
            self.pos = old_pos;
            result
        }

        fn read_enum_struct_variant_field<T>(&mut self,
                                             name: &str,
                                             idx: uint,
                                             f: &fn(&mut Decoder) -> T)
                                             -> T {
            debug!("read_enum_struct_variant_arg(name=%?, idx=%u)", name, idx);
            f(self)
        }

        fn read_struct<T>(&mut self,
                          name: &str,
                          _: uint,
                          f: &fn(&mut Decoder) -> T)
                          -> T {
            debug!("read_struct(name=%s)", name);
            f(self)
        }

        fn read_struct_field<T>(&mut self,
                                name: &str,
                                idx: uint,
                                f: &fn(&mut Decoder) -> T)
                                -> T {
            debug!("read_struct_field(name=%?, idx=%u)", name, idx);
            self._check_label(name);
            f(self)
        }

        fn read_tuple<T>(&mut self, f: &fn(&mut Decoder, uint) -> T) -> T {
            debug!("read_tuple()");
            self.read_seq(f)
        }

        fn read_tuple_arg<T>(&mut self, idx: uint, f: &fn(&mut Decoder) -> T)
                             -> T {
            debug!("read_tuple_arg(idx=%u)", idx);
            self.read_seq_elt(idx, f)
        }

        fn read_tuple_struct<T>(&mut self,
                                name: &str,
                                f: &fn(&mut Decoder, uint) -> T)
                                -> T {
            debug!("read_tuple_struct(name=%?)", name);
            self.read_tuple(f)
        }

        fn read_tuple_struct_arg<T>(&mut self,
                                    idx: uint,
                                    f: &fn(&mut Decoder) -> T)
                                    -> T {
            debug!("read_tuple_struct_arg(idx=%u)", idx);
            self.read_tuple_arg(idx, f)
        }

        fn read_option<T>(&mut self, f: &fn(&mut Decoder, bool) -> T) -> T {
            debug!("read_option()");
            do self.read_enum("Option") |this| {
                do this.read_enum_variant(["None", "Some"]) |this, idx| {
                    match idx {
                        0 => f(this, false),
                        1 => f(this, true),
                        _ => fail!(),
                    }
                }
            }
        }

        fn read_seq<T>(&mut self, f: &fn(&mut Decoder, uint) -> T) -> T {
            debug!("read_seq()");
            do self.push_doc(EsVec) |d| {
                let len = d._next_uint(EsVecLen);
                debug!("  len=%u", len);
                f(d, len)
            }
        }

        fn read_seq_elt<T>(&mut self, idx: uint, f: &fn(&mut Decoder) -> T)
                           -> T {
            debug!("read_seq_elt(idx=%u)", idx);
            self.push_doc(EsVecElt, f)
        }

        fn read_map<T>(&mut self, f: &fn(&mut Decoder, uint) -> T) -> T {
            debug!("read_map()");
            do self.push_doc(EsMap) |d| {
                let len = d._next_uint(EsMapLen);
                debug!("  len=%u", len);
                f(d, len)
            }
        }

        fn read_map_elt_key<T>(&mut self,
                               idx: uint,
                               f: &fn(&mut Decoder) -> T)
                               -> T {
            debug!("read_map_elt_key(idx=%u)", idx);
            self.push_doc(EsMapKey, f)
        }

        fn read_map_elt_val<T>(&mut self,
                               idx: uint,
                               f: &fn(&mut Decoder) -> T)
                               -> T {
            debug!("read_map_elt_val(idx=%u)", idx);
            self.push_doc(EsMapVal, f)
        }
    }
}

pub mod writer {
    use super::*;

    use core::cast;
    use core::io;
    use core::str;

    // ebml writing
    pub struct Encoder {
        writer: @io::Writer,
        priv size_positions: ~[uint],
    }

    fn write_sized_vuint(w: @io::Writer, n: uint, size: uint) {
        match size {
            1u => w.write(&[0x80u8 | (n as u8)]),
            2u => w.write(&[0x40u8 | ((n >> 8_u) as u8), n as u8]),
            3u => w.write(&[0x20u8 | ((n >> 16_u) as u8), (n >> 8_u) as u8,
                            n as u8]),
            4u => w.write(&[0x10u8 | ((n >> 24_u) as u8), (n >> 16_u) as u8,
                            (n >> 8_u) as u8, n as u8]),
            _ => fail!("vint to write too big: %?", n)
        };
    }

    fn write_vuint(w: @io::Writer, n: uint) {
        if n < 0x7f_u { write_sized_vuint(w, n, 1u); return; }
        if n < 0x4000_u { write_sized_vuint(w, n, 2u); return; }
        if n < 0x200000_u { write_sized_vuint(w, n, 3u); return; }
        if n < 0x10000000_u { write_sized_vuint(w, n, 4u); return; }
        fail!("vint to write too big: %?", n);
    }

    pub fn Encoder(w: @io::Writer) -> Encoder {
        let size_positions: ~[uint] = ~[];
        Encoder {
            writer: w,
            size_positions: size_positions
        }
    }

    // FIXME (#2741): Provide a function to write the standard ebml header.
    impl Encoder {
        pub fn start_tag(&mut self, tag_id: uint) {
            debug!("Start tag %u", tag_id);

            // Write the enum ID:
            write_vuint(self.writer, tag_id);

            // Write a placeholder four-byte size.
            self.size_positions.push(self.writer.tell());
            let zeroes: &[u8] = &[0u8, 0u8, 0u8, 0u8];
            self.writer.write(zeroes);
        }

        pub fn end_tag(&mut self) {
            let last_size_pos = self.size_positions.pop();
            let cur_pos = self.writer.tell();
            self.writer.seek(last_size_pos as int, io::SeekSet);
            let size = (cur_pos - last_size_pos - 4u);
            write_sized_vuint(self.writer, size, 4u);
            self.writer.seek(cur_pos as int, io::SeekSet);

            debug!("End tag (size = %u)", size);
        }

        pub fn wr_tag(&mut self, tag_id: uint, blk: &fn()) {
            self.start_tag(tag_id);
            blk();
            self.end_tag();
        }

        pub fn wr_tagged_bytes(&mut self, tag_id: uint, b: &[u8]) {
            write_vuint(self.writer, tag_id);
            write_vuint(self.writer, b.len());
            self.writer.write(b);
        }

        pub fn wr_tagged_u64(&mut self, tag_id: uint, v: u64) {
            do io::u64_to_be_bytes(v, 8u) |v| {
                self.wr_tagged_bytes(tag_id, v);
            }
        }

        pub fn wr_tagged_u32(&mut self, tag_id: uint, v: u32) {
            do io::u64_to_be_bytes(v as u64, 4u) |v| {
                self.wr_tagged_bytes(tag_id, v);
            }
        }

        pub fn wr_tagged_u16(&mut self, tag_id: uint, v: u16) {
            do io::u64_to_be_bytes(v as u64, 2u) |v| {
                self.wr_tagged_bytes(tag_id, v);
            }
        }

        pub fn wr_tagged_u8(&mut self, tag_id: uint, v: u8) {
            self.wr_tagged_bytes(tag_id, &[v]);
        }

        pub fn wr_tagged_i64(&mut self, tag_id: uint, v: i64) {
            do io::u64_to_be_bytes(v as u64, 8u) |v| {
                self.wr_tagged_bytes(tag_id, v);
            }
        }

        pub fn wr_tagged_i32(&mut self, tag_id: uint, v: i32) {
            do io::u64_to_be_bytes(v as u64, 4u) |v| {
                self.wr_tagged_bytes(tag_id, v);
            }
        }

        pub fn wr_tagged_i16(&mut self, tag_id: uint, v: i16) {
            do io::u64_to_be_bytes(v as u64, 2u) |v| {
                self.wr_tagged_bytes(tag_id, v);
            }
        }

        pub fn wr_tagged_i8(&mut self, tag_id: uint, v: i8) {
            self.wr_tagged_bytes(tag_id, &[v as u8]);
        }

        pub fn wr_tagged_str(&mut self, tag_id: uint, v: &str) {
            str::byte_slice(v, |b| self.wr_tagged_bytes(tag_id, b));
        }

        pub fn wr_bytes(&mut self, b: &[u8]) {
            debug!("Write %u bytes", b.len());
            self.writer.write(b);
        }

        pub fn wr_str(&mut self, s: &str) {
            debug!("Write str: %?", s);
            self.writer.write(str::to_bytes(s));
        }
    }

    // FIXME (#2743): optionally perform "relaxations" on end_tag to more
    // efficiently encode sizes; this is a fixed point iteration

    // Set to true to generate more debugging in EBML code.
    // Totally lame approach.
    static debug: bool = true;

    impl Encoder {
        // used internally to emit things like the vector length and so on
        fn _emit_tagged_uint(&mut self, t: EbmlEncoderTag, v: uint) {
            assert!(v <= 0xFFFF_FFFF_u);
            self.wr_tagged_u32(t as uint, v as u32);
        }

        fn _emit_label(&mut self, label: &str) {
            // There are various strings that we have access to, such as
            // the name of a record field, which do not actually appear in
            // the encoded EBML (normally).  This is just for
            // efficiency.  When debugging, though, we can emit such
            // labels and then they will be checked by decoder to
            // try and check failures more quickly.
            if debug { self.wr_tagged_str(EsLabel as uint, label) }
        }
    }

    impl Encoder {
        pub fn emit_opaque(&mut self, f: &fn(&mut Encoder)) {
            self.start_tag(EsOpaque as uint);
            f(self);
            self.end_tag();
        }
    }

    impl ::serialize::Encoder for Encoder {
        fn emit_nil(&mut self) {}

        fn emit_uint(&mut self, v: uint) {
            self.wr_tagged_u64(EsUint as uint, v as u64);
        }
        fn emit_u64(&mut self, v: u64) {
            self.wr_tagged_u64(EsU64 as uint, v);
        }
        fn emit_u32(&mut self, v: u32) {
            self.wr_tagged_u32(EsU32 as uint, v);
        }
        fn emit_u16(&mut self, v: u16) {
            self.wr_tagged_u16(EsU16 as uint, v);
        }
        fn emit_u8(&mut self, v: u8) {
            self.wr_tagged_u8(EsU8 as uint, v);
        }

        fn emit_int(&mut self, v: int) {
            self.wr_tagged_i64(EsInt as uint, v as i64);
        }
        fn emit_i64(&mut self, v: i64) {
            self.wr_tagged_i64(EsI64 as uint, v);
        }
        fn emit_i32(&mut self, v: i32) {
            self.wr_tagged_i32(EsI32 as uint, v);
        }
        fn emit_i16(&mut self, v: i16) {
            self.wr_tagged_i16(EsI16 as uint, v);
        }
        fn emit_i8(&mut self, v: i8) {
            self.wr_tagged_i8(EsI8 as uint, v);
        }

        fn emit_bool(&mut self, v: bool) {
            self.wr_tagged_u8(EsBool as uint, v as u8)
        }

        fn emit_f64(&mut self, v: f64) {
            let bits = unsafe { cast::transmute(v) };
            self.wr_tagged_u64(EsF64 as uint, bits);
        }
        fn emit_f32(&mut self, v: f32) {
            let bits = unsafe { cast::transmute(v) };
            self.wr_tagged_u32(EsF32 as uint, bits);
        }
        fn emit_float(&mut self, v: float) {
            let bits = unsafe { cast::transmute(v as f64) };
            self.wr_tagged_u64(EsFloat as uint, bits);
        }

        fn emit_char(&mut self, v: char) {
            self.wr_tagged_u32(EsChar as uint, v as u32);
        }

        fn emit_str(&mut self, v: &str) {
            self.wr_tagged_str(EsStr as uint, v)
        }

        fn emit_enum(&mut self, name: &str, f: &fn(&mut Encoder)) {
            self._emit_label(name);
            self.start_tag(EsEnum as uint);
            f(self);
            self.end_tag();
        }

        fn emit_enum_variant(&mut self,
                             _: &str,
                             v_id: uint,
                             _: uint,
                             f: &fn(&mut Encoder)) {
            self._emit_tagged_uint(EsEnumVid, v_id);
            self.start_tag(EsEnumBody as uint);
            f(self);
            self.end_tag();
        }

        fn emit_enum_variant_arg(&mut self, _: uint, f: &fn(&mut Encoder)) {
            f(self)
        }

        fn emit_enum_struct_variant(&mut self,
                                    v_name: &str,
                                    v_id: uint,
                                    cnt: uint,
                                    f: &fn(&mut Encoder)) {
            self.emit_enum_variant(v_name, v_id, cnt, f)
        }

        fn emit_enum_struct_variant_field(&mut self,
                                          _: &str,
                                          idx: uint,
                                          f: &fn(&mut Encoder)) {
            self.emit_enum_variant_arg(idx, f)
        }

        fn emit_struct(&mut self, _: &str, _len: uint, f: &fn(&mut Encoder)) {
            f(self)
        }

        fn emit_struct_field(&mut self,
                             name: &str,
                             _: uint,
                             f: &fn(&mut Encoder)) {
            self._emit_label(name);
            f(self)
        }

        fn emit_tuple(&mut self, len: uint, f: &fn(&mut Encoder)) {
            self.emit_seq(len, f)
        }
        fn emit_tuple_arg(&mut self, idx: uint, f: &fn(&mut Encoder)) {
            self.emit_seq_elt(idx, f)
        }

        fn emit_tuple_struct(&mut self,
                             _: &str,
                             len: uint,
                             f: &fn(&mut Encoder)) {
            self.emit_seq(len, f)
        }
        fn emit_tuple_struct_arg(&mut self, idx: uint, f: &fn(&mut Encoder)) {
            self.emit_seq_elt(idx, f)
        }

        fn emit_option(&mut self, f: &fn(&mut Encoder)) {
            self.emit_enum("Option", f);
        }
        fn emit_option_none(&mut self) {
            self.emit_enum_variant("None", 0, 0, |_| ())
        }
        fn emit_option_some(&mut self, f: &fn(&mut Encoder)) {
            self.emit_enum_variant("Some", 1, 1, f)
        }

        fn emit_seq(&mut self, len: uint, f: &fn(&mut Encoder)) {
            self.start_tag(EsVec as uint);
            self._emit_tagged_uint(EsVecLen, len);
            f(self);
            self.end_tag();
        }

        fn emit_seq_elt(&mut self, _idx: uint, f: &fn(&mut Encoder)) {
            self.start_tag(EsVecElt as uint);
            f(self);
            self.end_tag();
        }

        fn emit_map(&mut self, len: uint, f: &fn(&mut Encoder)) {
            self.start_tag(EsMap as uint);
            self._emit_tagged_uint(EsMapLen, len);
            f(self);
            self.end_tag();
        }

        fn emit_map_elt_key(&mut self, _idx: uint, f: &fn(&mut Encoder)) {
            self.start_tag(EsMapKey as uint);
            f(self);
            self.end_tag();
        }

        fn emit_map_elt_val(&mut self, _idx: uint, f: &fn(&mut Encoder)) {
            self.start_tag(EsMapVal as uint);
            f(self);
            self.end_tag();
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
                let mut ebml_w = writer::Encoder(wr);
                v.encode(&mut ebml_w)
            };
            let ebml_doc = reader::Doc(@bytes);
            let mut deser = reader::Decoder(ebml_doc);
            let v1 = serialize::Decodable::decode(&mut deser);
            debug!("v1 == %?", v1);
            assert_eq!(v, v1);
        }

        test_v(Some(22));
        test_v(None);
        test_v(Some(3));
    }
}
