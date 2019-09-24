use crate::leb128::{self, read_signed_leb128, write_signed_leb128};
use crate::serialize;
use std::borrow::Cow;

// -----------------------------------------------------------------------------
// Encoder
// -----------------------------------------------------------------------------

pub type EncodeResult = Result<(), !>;

pub struct Encoder {
    pub data: Vec<u8>,
}

impl Encoder {
    pub fn new(data: Vec<u8>) -> Encoder {
        Encoder { data }
    }

    pub fn into_inner(self) -> Vec<u8> {
        self.data
    }

    #[inline]
    pub fn emit_raw_bytes(&mut self, s: &[u8]) {
        self.data.extend_from_slice(s);
    }
}

macro_rules! write_uleb128 {
    ($enc:expr, $value:expr, $fun:ident) => {{
        leb128::$fun(&mut $enc.data, $value);
        Ok(())
    }}
}

macro_rules! write_sleb128 {
    ($enc:expr, $value:expr) => {{
        write_signed_leb128(&mut $enc.data, $value as i128);
        Ok(())
    }}
}

impl serialize::Encoder for Encoder {
    type Error = !;

    #[inline]
    fn emit_unit(&mut self) -> EncodeResult {
        Ok(())
    }

    #[inline]
    fn emit_usize(&mut self, v: usize) -> EncodeResult {
        write_uleb128!(self, v, write_usize_leb128)
    }

    #[inline]
    fn emit_u128(&mut self, v: u128) -> EncodeResult {
        write_uleb128!(self, v, write_u128_leb128)
    }

    #[inline]
    fn emit_u64(&mut self, v: u64) -> EncodeResult {
        write_uleb128!(self, v, write_u64_leb128)
    }

    #[inline]
    fn emit_u32(&mut self, v: u32) -> EncodeResult {
        write_uleb128!(self, v, write_u32_leb128)
    }

    #[inline]
    fn emit_u16(&mut self, v: u16) -> EncodeResult {
        write_uleb128!(self, v, write_u16_leb128)
    }

    #[inline]
    fn emit_u8(&mut self, v: u8) -> EncodeResult {
        self.data.push(v);
        Ok(())
    }

    #[inline]
    fn emit_isize(&mut self, v: isize) -> EncodeResult {
        write_sleb128!(self, v)
    }

    #[inline]
    fn emit_i128(&mut self, v: i128) -> EncodeResult {
        write_sleb128!(self, v)
    }

    #[inline]
    fn emit_i64(&mut self, v: i64) -> EncodeResult {
        write_sleb128!(self, v)
    }

    #[inline]
    fn emit_i32(&mut self, v: i32) -> EncodeResult {
        write_sleb128!(self, v)
    }

    #[inline]
    fn emit_i16(&mut self, v: i16) -> EncodeResult {
        write_sleb128!(self, v)
    }

    #[inline]
    fn emit_i8(&mut self, v: i8) -> EncodeResult {
        let as_u8: u8 = unsafe { ::std::mem::transmute(v) };
        self.emit_u8(as_u8)
    }

    #[inline]
    fn emit_bool(&mut self, v: bool) -> EncodeResult {
        self.emit_u8(if v {
            1
        } else {
            0
        })
    }

    #[inline]
    fn emit_f64(&mut self, v: f64) -> EncodeResult {
        let as_u64: u64 = unsafe { ::std::mem::transmute(v) };
        self.emit_u64(as_u64)
    }

    #[inline]
    fn emit_f32(&mut self, v: f32) -> EncodeResult {
        let as_u32: u32 = unsafe { ::std::mem::transmute(v) };
        self.emit_u32(as_u32)
    }

    #[inline]
    fn emit_char(&mut self, v: char) -> EncodeResult {
        self.emit_u32(v as u32)
    }

    #[inline]
    fn emit_str(&mut self, v: &str) -> EncodeResult {
        self.emit_usize(v.len())?;
        self.emit_raw_bytes(v.as_bytes());
        Ok(())
    }
}

impl Encoder {
    #[inline]
    pub fn position(&self) -> usize {
        self.data.len()
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
    #[inline]
    pub fn new(data: &'a [u8], position: usize) -> Decoder<'a> {
        Decoder {
            data,
            position,
        }
    }

    #[inline]
    pub fn position(&self) -> usize {
        self.position
    }

    #[inline]
    pub fn set_position(&mut self, pos: usize) {
        self.position = pos
    }

    #[inline]
    pub fn advance(&mut self, bytes: usize) {
        self.position += bytes;
    }

    #[inline]
    pub fn read_raw_bytes(&mut self, s: &mut [u8]) -> Result<(), String> {
        let start = self.position;
        let end = start + s.len();

        s.copy_from_slice(&self.data[start..end]);

        self.position = end;

        Ok(())
    }
}

macro_rules! read_uleb128 {
    ($dec:expr, $t:ty, $fun:ident) => ({
        let (value, bytes_read) = leb128::$fun(&$dec.data[$dec.position ..]);
        $dec.position += bytes_read;
        Ok(value)
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
        read_uleb128!(self, u128, read_u128_leb128)
    }

    #[inline]
    fn read_u64(&mut self) -> Result<u64, Self::Error> {
        read_uleb128!(self, u64, read_u64_leb128)
    }

    #[inline]
    fn read_u32(&mut self) -> Result<u32, Self::Error> {
        read_uleb128!(self, u32, read_u32_leb128)
    }

    #[inline]
    fn read_u16(&mut self) -> Result<u16, Self::Error> {
        read_uleb128!(self, u16, read_u16_leb128)
    }

    #[inline]
    fn read_u8(&mut self) -> Result<u8, Self::Error> {
        let value = self.data[self.position];
        self.position += 1;
        Ok(value)
    }

    #[inline]
    fn read_usize(&mut self) -> Result<usize, Self::Error> {
        read_uleb128!(self, usize, read_usize_leb128)
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
        Ok(f64::from_bits(bits))
    }

    #[inline]
    fn read_f32(&mut self) -> Result<f32, Self::Error> {
        let bits = self.read_u32()?;
        Ok(f32::from_bits(bits))
    }

    #[inline]
    fn read_char(&mut self) -> Result<char, Self::Error> {
        let bits = self.read_u32()?;
        Ok(::std::char::from_u32(bits).unwrap())
    }

    #[inline]
    fn read_str(&mut self) -> Result<Cow<'_, str>, Self::Error> {
        let len = self.read_usize()?;
        let s = ::std::str::from_utf8(&self.data[self.position..self.position + len]).unwrap();
        self.position += len;
        Ok(Cow::Borrowed(s))
    }

    #[inline]
    fn error(&mut self, err: &str) -> Self::Error {
        err.to_string()
    }
}
