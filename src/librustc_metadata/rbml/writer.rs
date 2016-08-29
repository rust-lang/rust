// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem;
use std::io::prelude::*;
use std::io::{self, SeekFrom, Cursor};

use rbml::opaque;
use rbml::reader::EbmlEncoderTag::*;
use rbml::reader::NUM_IMPLICIT_TAGS;

use rustc_serialize as serialize;

pub type EncodeResult = io::Result<()>;

// rbml writing
pub struct Encoder {
    pub writer: Cursor<Vec<u8>>,
    size_positions: Vec<u64>,
    relax_limit: u64, // do not move encoded bytes before this position
}

const NUM_TAGS: usize = 0x1000;

fn write_tag<W: Write>(w: &mut W, n: usize) -> EncodeResult {
    if n < 0xf0 {
        w.write_all(&[n as u8])
    } else if 0x100 <= n && n < NUM_TAGS {
        w.write_all(&[0xf0 | (n >> 8) as u8, n as u8])
    } else {
        Err(io::Error::new(io::ErrorKind::Other, &format!("invalid tag: {}", n)[..]))
    }
}

fn write_sized_vuint<W: Write>(w: &mut W, n: usize, size: usize) -> EncodeResult {
    match size {
        1 => w.write_all(&[0x80 | (n as u8)]),
        2 => w.write_all(&[0x40 | ((n >> 8) as u8), n as u8]),
        3 => w.write_all(&[0x20 | ((n >> 16) as u8), (n >> 8) as u8, n as u8]),
        4 => w.write_all(&[0x10 | ((n >> 24) as u8), (n >> 16) as u8, (n >> 8) as u8, n as u8]),
        _ => Err(io::Error::new(io::ErrorKind::Other, &format!("isize too big: {}", n)[..])),
    }
}

pub fn write_vuint<W: Write>(w: &mut W, n: usize) -> EncodeResult {
    if n < 0x7f {
        return write_sized_vuint(w, n, 1);
    }
    if n < 0x4000 {
        return write_sized_vuint(w, n, 2);
    }
    if n < 0x200000 {
        return write_sized_vuint(w, n, 3);
    }
    if n < 0x10000000 {
        return write_sized_vuint(w, n, 4);
    }
    Err(io::Error::new(io::ErrorKind::Other, &format!("isize too big: {}", n)[..]))
}

impl Encoder {
    pub fn new() -> Encoder {
        Encoder {
            writer: Cursor::new(vec![]),
            size_positions: vec![],
            relax_limit: 0,
        }
    }

    pub fn start_tag(&mut self, tag_id: usize) -> EncodeResult {
        debug!("Start tag {:?}", tag_id);
        assert!(tag_id >= NUM_IMPLICIT_TAGS);

        // Write the enum ID:
        write_tag(&mut self.writer, tag_id)?;

        // Write a placeholder four-byte size.
        let cur_pos = self.writer.seek(SeekFrom::Current(0))?;
        self.size_positions.push(cur_pos);
        let zeroes: &[u8] = &[0, 0, 0, 0];
        self.writer.write_all(zeroes)
    }

    pub fn end_tag(&mut self) -> EncodeResult {
        let last_size_pos = self.size_positions.pop().unwrap();
        let cur_pos = self.writer.seek(SeekFrom::Current(0))?;
        self.writer.seek(SeekFrom::Start(last_size_pos))?;
        let size = (cur_pos - last_size_pos - 4) as usize;

        // relax the size encoding for small tags (bigger tags are costly to move).
        // we should never try to move the stable positions, however.
        const RELAX_MAX_SIZE: usize = 0x100;
        if size <= RELAX_MAX_SIZE && last_size_pos >= self.relax_limit {
            // we can't alter the buffer in place, so have a temporary buffer
            let mut buf = [0u8; RELAX_MAX_SIZE];
            {
                let last_size_pos = last_size_pos as usize;
                let data = &self.writer.get_ref()[last_size_pos + 4..cur_pos as usize];
                buf[..size].copy_from_slice(data);
            }

            // overwrite the size and data and continue
            write_vuint(&mut self.writer, size)?;
            self.writer.write_all(&buf[..size])?;
        } else {
            // overwrite the size with an overlong encoding and skip past the data
            write_sized_vuint(&mut self.writer, size, 4)?;
            self.writer.seek(SeekFrom::Start(cur_pos))?;
        }

        debug!("End tag (size = {:?})", size);
        Ok(())
    }

    pub fn wr_tag<F>(&mut self, tag_id: usize, blk: F) -> EncodeResult
        where F: FnOnce() -> EncodeResult
    {
        self.start_tag(tag_id)?;
        blk()?;
        self.end_tag()
    }

    pub fn wr_tagged_bytes(&mut self, tag_id: usize, b: &[u8]) -> EncodeResult {
        assert!(tag_id >= NUM_IMPLICIT_TAGS);
        write_tag(&mut self.writer, tag_id)?;
        write_vuint(&mut self.writer, b.len())?;
        self.writer.write_all(b)
    }

    pub fn wr_tagged_u64(&mut self, tag_id: usize, v: u64) -> EncodeResult {
        let bytes: [u8; 8] = unsafe { mem::transmute(v.to_be()) };
        // tagged integers are emitted in big-endian, with no
        // leading zeros.
        let leading_zero_bytes = v.leading_zeros() / 8;
        self.wr_tagged_bytes(tag_id, &bytes[leading_zero_bytes as usize..])
    }

    #[inline]
    pub fn wr_tagged_u32(&mut self, tag_id: usize, v: u32) -> EncodeResult {
        self.wr_tagged_u64(tag_id, v as u64)
    }

    #[inline]
    pub fn wr_tagged_u16(&mut self, tag_id: usize, v: u16) -> EncodeResult {
        self.wr_tagged_u64(tag_id, v as u64)
    }

    #[inline]
    pub fn wr_tagged_u8(&mut self, tag_id: usize, v: u8) -> EncodeResult {
        self.wr_tagged_bytes(tag_id, &[v])
    }

    #[inline]
    pub fn wr_tagged_i64(&mut self, tag_id: usize, v: i64) -> EncodeResult {
        self.wr_tagged_u64(tag_id, v as u64)
    }

    #[inline]
    pub fn wr_tagged_i32(&mut self, tag_id: usize, v: i32) -> EncodeResult {
        self.wr_tagged_u32(tag_id, v as u32)
    }

    #[inline]
    pub fn wr_tagged_i16(&mut self, tag_id: usize, v: i16) -> EncodeResult {
        self.wr_tagged_u16(tag_id, v as u16)
    }

    #[inline]
    pub fn wr_tagged_i8(&mut self, tag_id: usize, v: i8) -> EncodeResult {
        self.wr_tagged_bytes(tag_id, &[v as u8])
    }

    pub fn wr_tagged_str(&mut self, tag_id: usize, v: &str) -> EncodeResult {
        self.wr_tagged_bytes(tag_id, v.as_bytes())
    }

    // for auto-serialization
    fn wr_tagged_raw_bytes(&mut self, tag_id: usize, b: &[u8]) -> EncodeResult {
        write_tag(&mut self.writer, tag_id)?;
        self.writer.write_all(b)
    }

    fn wr_tagged_raw_u64(&mut self, tag_id: usize, v: u64) -> EncodeResult {
        let bytes: [u8; 8] = unsafe { mem::transmute(v.to_be()) };
        self.wr_tagged_raw_bytes(tag_id, &bytes)
    }

    fn wr_tagged_raw_u32(&mut self, tag_id: usize, v: u32) -> EncodeResult {
        let bytes: [u8; 4] = unsafe { mem::transmute(v.to_be()) };
        self.wr_tagged_raw_bytes(tag_id, &bytes)
    }

    fn wr_tagged_raw_u16(&mut self, tag_id: usize, v: u16) -> EncodeResult {
        let bytes: [u8; 2] = unsafe { mem::transmute(v.to_be()) };
        self.wr_tagged_raw_bytes(tag_id, &bytes)
    }

    fn wr_tagged_raw_u8(&mut self, tag_id: usize, v: u8) -> EncodeResult {
        self.wr_tagged_raw_bytes(tag_id, &[v])
    }

    fn wr_tagged_raw_i64(&mut self, tag_id: usize, v: i64) -> EncodeResult {
        self.wr_tagged_raw_u64(tag_id, v as u64)
    }

    fn wr_tagged_raw_i32(&mut self, tag_id: usize, v: i32) -> EncodeResult {
        self.wr_tagged_raw_u32(tag_id, v as u32)
    }

    fn wr_tagged_raw_i16(&mut self, tag_id: usize, v: i16) -> EncodeResult {
        self.wr_tagged_raw_u16(tag_id, v as u16)
    }

    fn wr_tagged_raw_i8(&mut self, tag_id: usize, v: i8) -> EncodeResult {
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

    /// Returns the current position while marking it stable, i.e.
    /// generated bytes so far wouldn't be affected by relaxation.
    pub fn mark_stable_position(&mut self) -> u64 {
        let pos = self.writer.seek(SeekFrom::Current(0)).unwrap();
        if self.relax_limit < pos {
            self.relax_limit = pos;
        }
        pos
    }
}

impl Encoder {
    // used internally to emit things like the vector length and so on
    fn _emit_tagged_sub(&mut self, v: usize) -> EncodeResult {
        if v as u8 as usize == v {
            self.wr_tagged_raw_u8(EsSub8 as usize, v as u8)
        } else if v as u32 as usize == v {
            self.wr_tagged_raw_u32(EsSub32 as usize, v as u32)
        } else {
            Err(io::Error::new(io::ErrorKind::Other,
                               &format!("length or variant id too big: {}", v)[..]))
        }
    }

    pub fn emit_opaque<F>(&mut self, f: F) -> EncodeResult
        where F: FnOnce(&mut opaque::Encoder) -> EncodeResult
    {
        self.start_tag(EsOpaque as usize)?;

        {
            let mut opaque_encoder = opaque::Encoder::new(&mut self.writer);
            f(&mut opaque_encoder)?;
        }

        self.mark_stable_position();
        self.end_tag()
    }
}

impl<'a, 'tcx> serialize::Encoder for ::encoder::EncodeContext<'a, 'tcx> {
    type Error = io::Error;

    fn emit_nil(&mut self) -> EncodeResult {
        Ok(())
    }

    fn emit_usize(&mut self, v: usize) -> EncodeResult {
        self.emit_u64(v as u64)
    }
    fn emit_u64(&mut self, v: u64) -> EncodeResult {
        if v as u32 as u64 == v {
            self.emit_u32(v as u32)
        } else {
            self.wr_tagged_raw_u64(EsU64 as usize, v)
        }
    }
    fn emit_u32(&mut self, v: u32) -> EncodeResult {
        if v as u16 as u32 == v {
            self.emit_u16(v as u16)
        } else {
            self.wr_tagged_raw_u32(EsU32 as usize, v)
        }
    }
    fn emit_u16(&mut self, v: u16) -> EncodeResult {
        if v as u8 as u16 == v {
            self.emit_u8(v as u8)
        } else {
            self.wr_tagged_raw_u16(EsU16 as usize, v)
        }
    }
    fn emit_u8(&mut self, v: u8) -> EncodeResult {
        self.wr_tagged_raw_u8(EsU8 as usize, v)
    }

    fn emit_isize(&mut self, v: isize) -> EncodeResult {
        self.emit_i64(v as i64)
    }
    fn emit_i64(&mut self, v: i64) -> EncodeResult {
        if v as i32 as i64 == v {
            self.emit_i32(v as i32)
        } else {
            self.wr_tagged_raw_i64(EsI64 as usize, v)
        }
    }
    fn emit_i32(&mut self, v: i32) -> EncodeResult {
        if v as i16 as i32 == v {
            self.emit_i16(v as i16)
        } else {
            self.wr_tagged_raw_i32(EsI32 as usize, v)
        }
    }
    fn emit_i16(&mut self, v: i16) -> EncodeResult {
        if v as i8 as i16 == v {
            self.emit_i8(v as i8)
        } else {
            self.wr_tagged_raw_i16(EsI16 as usize, v)
        }
    }
    fn emit_i8(&mut self, v: i8) -> EncodeResult {
        self.wr_tagged_raw_i8(EsI8 as usize, v)
    }

    fn emit_bool(&mut self, v: bool) -> EncodeResult {
        self.wr_tagged_raw_u8(EsBool as usize, v as u8)
    }

    fn emit_f64(&mut self, v: f64) -> EncodeResult {
        let bits = unsafe { mem::transmute(v) };
        self.wr_tagged_raw_u64(EsF64 as usize, bits)
    }
    fn emit_f32(&mut self, v: f32) -> EncodeResult {
        let bits = unsafe { mem::transmute(v) };
        self.wr_tagged_raw_u32(EsF32 as usize, bits)
    }
    fn emit_char(&mut self, v: char) -> EncodeResult {
        self.wr_tagged_raw_u32(EsChar as usize, v as u32)
    }

    fn emit_str(&mut self, v: &str) -> EncodeResult {
        self.wr_tagged_str(EsStr as usize, v)
    }

    fn emit_enum<F>(&mut self, _name: &str, f: F) -> EncodeResult
        where F: FnOnce(&mut Self) -> EncodeResult
    {
        self.start_tag(EsEnum as usize)?;
        f(self)?;
        self.end_tag()
    }

    fn emit_enum_variant<F>(&mut self, _: &str, v_id: usize, _: usize, f: F) -> EncodeResult
        where F: FnOnce(&mut Self) -> EncodeResult
    {
        self._emit_tagged_sub(v_id)?;
        f(self)
    }

    fn emit_enum_variant_arg<F>(&mut self, _: usize, f: F) -> EncodeResult
        where F: FnOnce(&mut Self) -> EncodeResult
    {
        f(self)
    }

    fn emit_enum_struct_variant<F>(&mut self,
                                   v_name: &str,
                                   v_id: usize,
                                   cnt: usize,
                                   f: F)
                                   -> EncodeResult
        where F: FnOnce(&mut Self) -> EncodeResult
    {
        self.emit_enum_variant(v_name, v_id, cnt, f)
    }

    fn emit_enum_struct_variant_field<F>(&mut self, _: &str, idx: usize, f: F) -> EncodeResult
        where F: FnOnce(&mut Self) -> EncodeResult
    {
        self.emit_enum_variant_arg(idx, f)
    }

    fn emit_struct<F>(&mut self, _: &str, _len: usize, f: F) -> EncodeResult
        where F: FnOnce(&mut Self) -> EncodeResult
    {
        f(self)
    }

    fn emit_struct_field<F>(&mut self, _name: &str, _: usize, f: F) -> EncodeResult
        where F: FnOnce(&mut Self) -> EncodeResult
    {
        f(self)
    }

    fn emit_tuple<F>(&mut self, len: usize, f: F) -> EncodeResult
        where F: FnOnce(&mut Self) -> EncodeResult
    {
        self.emit_seq(len, f)
    }
    fn emit_tuple_arg<F>(&mut self, idx: usize, f: F) -> EncodeResult
        where F: FnOnce(&mut Self) -> EncodeResult
    {
        self.emit_seq_elt(idx, f)
    }

    fn emit_tuple_struct<F>(&mut self, _: &str, len: usize, f: F) -> EncodeResult
        where F: FnOnce(&mut Self) -> EncodeResult
    {
        self.emit_seq(len, f)
    }
    fn emit_tuple_struct_arg<F>(&mut self, idx: usize, f: F) -> EncodeResult
        where F: FnOnce(&mut Self) -> EncodeResult
    {
        self.emit_seq_elt(idx, f)
    }

    fn emit_option<F>(&mut self, f: F) -> EncodeResult
        where F: FnOnce(&mut Self) -> EncodeResult
    {
        self.emit_enum("Option", f)
    }
    fn emit_option_none(&mut self) -> EncodeResult {
        self.emit_enum_variant("None", 0, 0, |_| Ok(()))
    }
    fn emit_option_some<F>(&mut self, f: F) -> EncodeResult
        where F: FnOnce(&mut Self) -> EncodeResult
    {

        self.emit_enum_variant("Some", 1, 1, f)
    }

    fn emit_seq<F>(&mut self, len: usize, f: F) -> EncodeResult
        where F: FnOnce(&mut Self) -> EncodeResult
    {
        if len == 0 {
            // empty vector optimization
            return self.wr_tagged_bytes(EsVec as usize, &[]);
        }

        self.start_tag(EsVec as usize)?;
        self._emit_tagged_sub(len)?;
        f(self)?;
        self.end_tag()
    }

    fn emit_seq_elt<F>(&mut self, _idx: usize, f: F) -> EncodeResult
        where F: FnOnce(&mut Self) -> EncodeResult
    {

        self.start_tag(EsVecElt as usize)?;
        f(self)?;
        self.end_tag()
    }

    fn emit_map<F>(&mut self, len: usize, f: F) -> EncodeResult
        where F: FnOnce(&mut Self) -> EncodeResult
    {
        if len == 0 {
            // empty map optimization
            return self.wr_tagged_bytes(EsMap as usize, &[]);
        }

        self.start_tag(EsMap as usize)?;
        self._emit_tagged_sub(len)?;
        f(self)?;
        self.end_tag()
    }

    fn emit_map_elt_key<F>(&mut self, _idx: usize, f: F) -> EncodeResult
        where F: FnOnce(&mut Self) -> EncodeResult
    {

        self.start_tag(EsMapKey as usize)?;
        f(self)?;
        self.end_tag()
    }

    fn emit_map_elt_val<F>(&mut self, _idx: usize, f: F) -> EncodeResult
        where F: FnOnce(&mut Self) -> EncodeResult
    {
        self.start_tag(EsMapVal as usize)?;
        f(self)?;
        self.end_tag()
    }
}
