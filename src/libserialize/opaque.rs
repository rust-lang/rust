// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use leb128::{read_signed_leb128, read_unsigned_leb128, write_signed_leb128, write_unsigned_leb128};
use std::borrow::Cow;
use std::io::{self, Write};
use serialize;

use rustc_i128::{i128, u128};

// -----------------------------------------------------------------------------
// Encoder
// -----------------------------------------------------------------------------

pub type EncodeResult = io::Result<()>;

pub struct Encoder<'a> {
    pub cursor: &'a mut io::Cursor<Vec<u8>>,
}

impl<'a> Encoder<'a> {
    pub fn new(cursor: &'a mut io::Cursor<Vec<u8>>) -> Encoder<'a> {
        Encoder { cursor: cursor }
    }
}


macro_rules! write_uleb128 {
    ($enc:expr, $value:expr) => {{
        let pos = $enc.cursor.position() as usize;
        let bytes_written = write_unsigned_leb128($enc.cursor.get_mut(), pos, $value as u128);
        $enc.cursor.set_position((pos + bytes_written) as u64);
        Ok(())
    }}
}

macro_rules! write_sleb128 {
    ($enc:expr, $value:expr) => {{
        let pos = $enc.cursor.position() as usize;
        let bytes_written = write_signed_leb128($enc.cursor.get_mut(), pos, $value as i128);
        $enc.cursor.set_position((pos + bytes_written) as u64);
        Ok(())
    }}
}

impl<'a> serialize::Encoder for Encoder<'a> {
    type Error = io::Error;

    fn emit_nil(&mut self) -> EncodeResult {
        Ok(())
    }

    fn emit_usize(&mut self, v: usize) -> EncodeResult {
        write_uleb128!(self, v)
    }

    fn emit_u128(&mut self, v: u128) -> EncodeResult {
        write_uleb128!(self, v)
    }

    fn emit_u64(&mut self, v: u64) -> EncodeResult {
        write_uleb128!(self, v)
    }

    fn emit_u32(&mut self, v: u32) -> EncodeResult {
        write_uleb128!(self, v)
    }

    fn emit_u16(&mut self, v: u16) -> EncodeResult {
        write_uleb128!(self, v)
    }

    fn emit_u8(&mut self, v: u8) -> EncodeResult {
        let _ = self.cursor.write_all(&[v]);
        Ok(())
    }

    fn emit_isize(&mut self, v: isize) -> EncodeResult {
        write_sleb128!(self, v)
    }

    fn emit_i128(&mut self, v: i128) -> EncodeResult {
        write_sleb128!(self, v)
    }

    fn emit_i64(&mut self, v: i64) -> EncodeResult {
        write_sleb128!(self, v)
    }

    fn emit_i32(&mut self, v: i32) -> EncodeResult {
        write_sleb128!(self, v)
    }

    fn emit_i16(&mut self, v: i16) -> EncodeResult {
        write_sleb128!(self, v)
    }

    fn emit_i8(&mut self, v: i8) -> EncodeResult {
        let as_u8: u8 = unsafe { ::std::mem::transmute(v) };
        let _ = self.cursor.write_all(&[as_u8]);
        Ok(())
    }

    fn emit_bool(&mut self, v: bool) -> EncodeResult {
        self.emit_u8(if v {
            1
        } else {
            0
        })
    }

    fn emit_f64(&mut self, v: f64) -> EncodeResult {
        let as_u64: u64 = unsafe { ::std::mem::transmute(v) };
        self.emit_u64(as_u64)
    }

    fn emit_f32(&mut self, v: f32) -> EncodeResult {
        let as_u32: u32 = unsafe { ::std::mem::transmute(v) };
        self.emit_u32(as_u32)
    }

    fn emit_char(&mut self, v: char) -> EncodeResult {
        self.emit_u32(v as u32)
    }

    fn emit_str(&mut self, v: &str) -> EncodeResult {
        self.emit_usize(v.len())?;
        let _ = self.cursor.write_all(v.as_bytes());
        Ok(())
    }
}

impl<'a> Encoder<'a> {
    pub fn position(&self) -> usize {
        self.cursor.position() as usize
    }
}

// -----------------------------------------------------------------------------
// Decoder
// -----------------------------------------------------------------------------

pub struct Decoder<'a> {
    pub data: &'a [u8],
    position: usize,
}

impl<'a> Decoder<'a> {
    pub fn new(data: &'a [u8], position: usize) -> Decoder<'a> {
        Decoder {
            data: data,
            position: position,
        }
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn advance(&mut self, bytes: usize) {
        self.position += bytes;
    }
}

macro_rules! read_uleb128 {
    ($dec:expr, $t:ty) => ({
        let (value, bytes_read) = read_unsigned_leb128($dec.data, $dec.position);
        $dec.position += bytes_read;
        Ok(value as $t)
    })
}

macro_rules! read_sleb128 {
    ($dec:expr, $t:ty) => ({
        let (value, bytes_read) = read_signed_leb128($dec.data, $dec.position);
        $dec.position += bytes_read;
        Ok(value as $t)
    })
}


impl<'a> serialize::Decoder for Decoder<'a> {
    type Error = String;

    #[inline]
    fn read_nil(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    #[inline]
    fn read_u128(&mut self) -> Result<u128, Self::Error> {
        read_uleb128!(self, u128)
    }

    #[inline]
    fn read_u64(&mut self) -> Result<u64, Self::Error> {
        read_uleb128!(self, u64)
    }

    #[inline]
    fn read_u32(&mut self) -> Result<u32, Self::Error> {
        read_uleb128!(self, u32)
    }

    #[inline]
    fn read_u16(&mut self) -> Result<u16, Self::Error> {
        read_uleb128!(self, u16)
    }

    #[inline]
    fn read_u8(&mut self) -> Result<u8, Self::Error> {
        let value = self.data[self.position];
        self.position += 1;
        Ok(value)
    }

    #[inline]
    fn read_usize(&mut self) -> Result<usize, Self::Error> {
        read_uleb128!(self, usize)
    }

    #[inline]
    fn read_i128(&mut self) -> Result<i128, Self::Error> {
        read_sleb128!(self, i128)
    }

    #[inline]
    fn read_i64(&mut self) -> Result<i64, Self::Error> {
        read_sleb128!(self, i64)
    }

    #[inline]
    fn read_i32(&mut self) -> Result<i32, Self::Error> {
        read_sleb128!(self, i32)
    }

    #[inline]
    fn read_i16(&mut self) -> Result<i16, Self::Error> {
        read_sleb128!(self, i16)
    }

    #[inline]
    fn read_i8(&mut self) -> Result<i8, Self::Error> {
        let as_u8 = self.data[self.position];
        self.position += 1;
        unsafe { Ok(::std::mem::transmute(as_u8)) }
    }

    #[inline]
    fn read_isize(&mut self) -> Result<isize, Self::Error> {
        read_sleb128!(self, isize)
    }

    #[inline]
    fn read_bool(&mut self) -> Result<bool, Self::Error> {
        let value = self.read_u8()?;
        Ok(value != 0)
    }

    #[inline]
    fn read_f64(&mut self) -> Result<f64, Self::Error> {
        let bits = self.read_u64()?;
        Ok(unsafe { ::std::mem::transmute(bits) })
    }

    #[inline]
    fn read_f32(&mut self) -> Result<f32, Self::Error> {
        let bits = self.read_u32()?;
        Ok(unsafe { ::std::mem::transmute(bits) })
    }

    #[inline]
    fn read_char(&mut self) -> Result<char, Self::Error> {
        let bits = self.read_u32()?;
        Ok(::std::char::from_u32(bits).unwrap())
    }

    #[inline]
    fn read_str(&mut self) -> Result<Cow<str>, Self::Error> {
        let len = self.read_usize()?;
        let s = ::std::str::from_utf8(&self.data[self.position..self.position + len]).unwrap();
        self.position += len;
        Ok(Cow::Borrowed(s))
    }

    fn error(&mut self, err: &str) -> Self::Error {
        err.to_string()
    }
}


#[cfg(test)]
mod tests {
    use serialize::{Encodable, Decodable};
    use std::io::Cursor;
    use std::fmt::Debug;
    use super::{Encoder, Decoder};

    #[derive(PartialEq, Clone, Debug, RustcEncodable, RustcDecodable)]
    struct Struct {
        a: (),
        b: u8,
        c: u16,
        d: u32,
        e: u64,
        f: usize,

        g: i8,
        h: i16,
        i: i32,
        j: i64,
        k: isize,

        l: char,
        m: String,
        n: f32,
        o: f64,
        p: bool,
        q: Option<u32>,
    }


    fn check_round_trip<T: Encodable + Decodable + PartialEq + Debug>(values: Vec<T>) {
        let mut cursor = Cursor::new(Vec::new());

        for value in &values {
            let mut encoder = Encoder::new(&mut cursor);
            Encodable::encode(&value, &mut encoder).unwrap();
        }

        let data = cursor.into_inner();
        let mut decoder = Decoder::new(&data[..], 0);

        for value in values {
            let decoded = Decodable::decode(&mut decoder).unwrap();
            assert_eq!(value, decoded);
        }
    }

    #[test]
    fn test_unit() {
        check_round_trip(vec![(), (), (), ()]);
    }

    #[test]
    fn test_u8() {
        let mut vec = vec![];
        for i in ::std::u8::MIN..::std::u8::MAX {
            vec.push(i);
        }
        check_round_trip(vec);
    }

    #[test]
    fn test_u16() {
        for i in ::std::u16::MIN..::std::u16::MAX {
            check_round_trip(vec![1, 2, 3, i, i, i]);
        }
    }

    #[test]
    fn test_u32() {
        check_round_trip(vec![1, 2, 3, ::std::u32::MIN, 0, 1, ::std::u32::MAX, 2, 1]);
    }

    #[test]
    fn test_u64() {
        check_round_trip(vec![1, 2, 3, ::std::u64::MIN, 0, 1, ::std::u64::MAX, 2, 1]);
    }

    #[test]
    fn test_usize() {
        check_round_trip(vec![1, 2, 3, ::std::usize::MIN, 0, 1, ::std::usize::MAX, 2, 1]);
    }

    #[test]
    fn test_i8() {
        let mut vec = vec![];
        for i in ::std::i8::MIN..::std::i8::MAX {
            vec.push(i);
        }
        check_round_trip(vec);
    }

    #[test]
    fn test_i16() {
        for i in ::std::i16::MIN..::std::i16::MAX {
            check_round_trip(vec![-1, 2, -3, i, i, i, 2]);
        }
    }

    #[test]
    fn test_i32() {
        check_round_trip(vec![-1, 2, -3, ::std::i32::MIN, 0, 1, ::std::i32::MAX, 2, 1]);
    }

    #[test]
    fn test_i64() {
        check_round_trip(vec![-1, 2, -3, ::std::i64::MIN, 0, 1, ::std::i64::MAX, 2, 1]);
    }

    #[test]
    fn test_isize() {
        check_round_trip(vec![-1, 2, -3, ::std::isize::MIN, 0, 1, ::std::isize::MAX, 2, 1]);
    }

    #[test]
    fn test_bool() {
        check_round_trip(vec![false, true, true, false, false]);
    }

    #[test]
    fn test_f32() {
        let mut vec = vec![];
        for i in -100..100 {
            vec.push((i as f32) / 3.0);
        }
        check_round_trip(vec);
    }

    #[test]
    fn test_f64() {
        let mut vec = vec![];
        for i in -100..100 {
            vec.push((i as f64) / 3.0);
        }
        check_round_trip(vec);
    }

    #[test]
    fn test_char() {
        let vec = vec!['a', 'b', 'c', 'd', 'A', 'X', ' ', '#', 'Ö', 'Ä', 'µ', '€'];
        check_round_trip(vec);
    }

    #[test]
    fn test_string() {
        let vec = vec!["abcbuÖeiovÄnameÜavmpßvmea€µsbpnvapeapmaebn".to_string(),
                       "abcbuÖganeiovÄnameÜavmpßvmea€µsbpnvapeapmaebn".to_string(),
                       "abcbuÖganeiovÄnameÜavmpßvmea€µsbpapmaebn".to_string(),
                       "abcbuÖganeiovÄnameÜavmpßvmeabpnvapeapmaebn".to_string(),
                       "abcbuÖganeiÄnameÜavmpßvmea€µsbpnvapeapmaebn".to_string(),
                       "abcbuÖganeiovÄnameÜavmpßvmea€µsbpmaebn".to_string(),
                       "abcbuÖganeiovÄnameÜavmpßvmea€µnvapeapmaebn".to_string()];

        check_round_trip(vec);
    }

    #[test]
    fn test_option() {
        check_round_trip(vec![Some(-1i8)]);
        check_round_trip(vec![Some(-2i16)]);
        check_round_trip(vec![Some(-3i32)]);
        check_round_trip(vec![Some(-4i64)]);
        check_round_trip(vec![Some(-5isize)]);

        let none_i8: Option<i8> = None;
        check_round_trip(vec![none_i8]);

        let none_i16: Option<i16> = None;
        check_round_trip(vec![none_i16]);

        let none_i32: Option<i32> = None;
        check_round_trip(vec![none_i32]);

        let none_i64: Option<i64> = None;
        check_round_trip(vec![none_i64]);

        let none_isize: Option<isize> = None;
        check_round_trip(vec![none_isize]);
    }

    #[test]
    fn test_struct() {
        check_round_trip(vec![Struct {
                                  a: (),
                                  b: 10,
                                  c: 11,
                                  d: 12,
                                  e: 13,
                                  f: 14,

                                  g: 15,
                                  h: 16,
                                  i: 17,
                                  j: 18,
                                  k: 19,

                                  l: 'x',
                                  m: "abc".to_string(),
                                  n: 20.5,
                                  o: 21.5,
                                  p: false,
                                  q: None,
                              }]);

        check_round_trip(vec![Struct {
                                  a: (),
                                  b: 101,
                                  c: 111,
                                  d: 121,
                                  e: 131,
                                  f: 141,

                                  g: -15,
                                  h: -16,
                                  i: -17,
                                  j: -18,
                                  k: -19,

                                  l: 'y',
                                  m: "def".to_string(),
                                  n: -20.5,
                                  o: -21.5,
                                  p: true,
                                  q: Some(1234567),
                              }]);
    }

    #[derive(PartialEq, Clone, Debug, RustcEncodable, RustcDecodable)]
    enum Enum {
        Variant1,
        Variant2(usize, f32),
        Variant3 {
            a: i32,
            b: char,
            c: bool,
        },
    }

    #[test]
    fn test_enum() {
        check_round_trip(vec![Enum::Variant1,
                              Enum::Variant2(1, 2.5),
                              Enum::Variant3 {
                                  a: 3,
                                  b: 'b',
                                  c: false,
                              },
                              Enum::Variant3 {
                                  a: -4,
                                  b: 'f',
                                  c: true,
                              }]);
    }

    #[test]
    fn test_sequence() {
        let mut vec = vec![];
        for i in -100i64..100i64 {
            vec.push(i * 100000);
        }

        check_round_trip(vec![vec]);
    }

    #[test]
    fn test_hash_map() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        for i in -100i64..100i64 {
            map.insert(i * 100000, i * 10000);
        }

        check_round_trip(vec![map]);
    }

    #[test]
    fn test_tuples() {
        check_round_trip(vec![('x', (), false, 0.5f32)]);
        check_round_trip(vec![(9i8, 10u16, 1.5f64)]);
        check_round_trip(vec![(-12i16, 11u8, 12usize)]);
        check_round_trip(vec![(1234567isize, 100000000000000u64, 99999999999999i64)]);
        check_round_trip(vec![(String::new(), "some string".to_string())]);
    }
}
