use crate::leb128::{self, max_leb128_len};
use crate::serialize::{self, Encoder as _};
use std::borrow::Cow;
use std::convert::TryInto;
use std::fs::File;
use std::io::{self, Write};
use std::mem::MaybeUninit;
use std::path::Path;
use std::ptr;

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
    pub fn position(&self) -> usize {
        self.data.len()
    }
}

macro_rules! write_leb128 {
    ($enc:expr, $value:expr, $int_ty:ty, $fun:ident) => {{
        const MAX_ENCODED_LEN: usize = max_leb128_len!($int_ty);
        let old_len = $enc.data.len();

        if MAX_ENCODED_LEN > $enc.data.capacity() - old_len {
            $enc.data.reserve(MAX_ENCODED_LEN);
        }

        // SAFETY: The above check and `reserve` ensures that there is enough
        // room to write the encoded value to the vector's internal buffer.
        unsafe {
            let buf = &mut *($enc.data.as_mut_ptr().add(old_len)
                as *mut [MaybeUninit<u8>; MAX_ENCODED_LEN]);
            let encoded = leb128::$fun(buf, $value);
            $enc.data.set_len(old_len + encoded.len());
        }

        Ok(())
    }};
}

/// A byte that [cannot occur in UTF8 sequences][utf8]. Used to mark the end of a string.
/// This way we can skip validation and still be relatively sure that deserialization
/// did not desynchronize.
///
/// [utf8]: https://en.wikipedia.org/w/index.php?title=UTF-8&oldid=1058865525#Codepage_layout
const STR_SENTINEL: u8 = 0xC1;

impl serialize::Encoder for Encoder {
    type Error = !;

    #[inline]
    fn emit_unit(&mut self) -> EncodeResult {
        Ok(())
    }

    #[inline]
    fn emit_usize(&mut self, v: usize) -> EncodeResult {
        write_leb128!(self, v, usize, write_usize_leb128)
    }

    #[inline]
    fn emit_u128(&mut self, v: u128) -> EncodeResult {
        write_leb128!(self, v, u128, write_u128_leb128)
    }

    #[inline]
    fn emit_u64(&mut self, v: u64) -> EncodeResult {
        write_leb128!(self, v, u64, write_u64_leb128)
    }

    #[inline]
    fn emit_u32(&mut self, v: u32) -> EncodeResult {
        write_leb128!(self, v, u32, write_u32_leb128)
    }

    #[inline]
    fn emit_u16(&mut self, v: u16) -> EncodeResult {
        self.data.extend_from_slice(&v.to_le_bytes());
        Ok(())
    }

    #[inline]
    fn emit_u8(&mut self, v: u8) -> EncodeResult {
        self.data.push(v);
        Ok(())
    }

    #[inline]
    fn emit_isize(&mut self, v: isize) -> EncodeResult {
        write_leb128!(self, v, isize, write_isize_leb128)
    }

    #[inline]
    fn emit_i128(&mut self, v: i128) -> EncodeResult {
        write_leb128!(self, v, i128, write_i128_leb128)
    }

    #[inline]
    fn emit_i64(&mut self, v: i64) -> EncodeResult {
        write_leb128!(self, v, i64, write_i64_leb128)
    }

    #[inline]
    fn emit_i32(&mut self, v: i32) -> EncodeResult {
        write_leb128!(self, v, i32, write_i32_leb128)
    }

    #[inline]
    fn emit_i16(&mut self, v: i16) -> EncodeResult {
        self.data.extend_from_slice(&v.to_le_bytes());
        Ok(())
    }

    #[inline]
    fn emit_i8(&mut self, v: i8) -> EncodeResult {
        let as_u8: u8 = unsafe { std::mem::transmute(v) };
        self.emit_u8(as_u8)
    }

    #[inline]
    fn emit_bool(&mut self, v: bool) -> EncodeResult {
        self.emit_u8(if v { 1 } else { 0 })
    }

    #[inline]
    fn emit_f64(&mut self, v: f64) -> EncodeResult {
        let as_u64: u64 = v.to_bits();
        self.emit_u64(as_u64)
    }

    #[inline]
    fn emit_f32(&mut self, v: f32) -> EncodeResult {
        let as_u32: u32 = v.to_bits();
        self.emit_u32(as_u32)
    }

    #[inline]
    fn emit_char(&mut self, v: char) -> EncodeResult {
        self.emit_u32(v as u32)
    }

    #[inline]
    fn emit_str(&mut self, v: &str) -> EncodeResult {
        self.emit_usize(v.len())?;
        self.emit_raw_bytes(v.as_bytes())?;
        self.emit_u8(STR_SENTINEL)
    }

    #[inline]
    fn emit_raw_bytes(&mut self, s: &[u8]) -> EncodeResult {
        self.data.extend_from_slice(s);
        Ok(())
    }
}

pub type FileEncodeResult = Result<(), io::Error>;

// `FileEncoder` encodes data to file via fixed-size buffer.
//
// When encoding large amounts of data to a file, using `FileEncoder` may be
// preferred over using `Encoder` to encode to a `Vec`, and then writing the
// `Vec` to file, as the latter uses as much memory as there is encoded data,
// while the former uses the fixed amount of memory allocated to the buffer.
// `FileEncoder` also has the advantage of not needing to reallocate as data
// is appended to it, but the disadvantage of requiring more error handling,
// which has some runtime overhead.
pub struct FileEncoder {
    // The input buffer. For adequate performance, we need more control over
    // buffering than `BufWriter` offers. If `BufWriter` ever offers a raw
    // buffer access API, we can use it, and remove `buf` and `buffered`.
    buf: Box<[MaybeUninit<u8>]>,
    buffered: usize,
    flushed: usize,
    file: File,
}

impl FileEncoder {
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        const DEFAULT_BUF_SIZE: usize = 8192;
        FileEncoder::with_capacity(path, DEFAULT_BUF_SIZE)
    }

    pub fn with_capacity<P: AsRef<Path>>(path: P, capacity: usize) -> io::Result<Self> {
        // Require capacity at least as large as the largest LEB128 encoding
        // here, so that we don't have to check or handle this on every write.
        assert!(capacity >= max_leb128_len());

        // Require capacity small enough such that some capacity checks can be
        // done using guaranteed non-overflowing add rather than sub, which
        // shaves an instruction off those code paths (on x86 at least).
        assert!(capacity <= usize::MAX - max_leb128_len());

        let file = File::create(path)?;

        Ok(FileEncoder { buf: Box::new_uninit_slice(capacity), buffered: 0, flushed: 0, file })
    }

    #[inline]
    pub fn position(&self) -> usize {
        // Tracking position this way instead of having a `self.position` field
        // means that we don't have to update the position on every write call.
        self.flushed + self.buffered
    }

    pub fn flush(&mut self) -> FileEncodeResult {
        // This is basically a copy of `BufWriter::flush`. If `BufWriter` ever
        // offers a raw buffer access API, we can use it, and remove this.

        /// Helper struct to ensure the buffer is updated after all the writes
        /// are complete. It tracks the number of written bytes and drains them
        /// all from the front of the buffer when dropped.
        struct BufGuard<'a> {
            buffer: &'a mut [u8],
            encoder_buffered: &'a mut usize,
            encoder_flushed: &'a mut usize,
            flushed: usize,
        }

        impl<'a> BufGuard<'a> {
            fn new(
                buffer: &'a mut [u8],
                encoder_buffered: &'a mut usize,
                encoder_flushed: &'a mut usize,
            ) -> Self {
                assert_eq!(buffer.len(), *encoder_buffered);
                Self { buffer, encoder_buffered, encoder_flushed, flushed: 0 }
            }

            /// The unwritten part of the buffer
            fn remaining(&self) -> &[u8] {
                &self.buffer[self.flushed..]
            }

            /// Flag some bytes as removed from the front of the buffer
            fn consume(&mut self, amt: usize) {
                self.flushed += amt;
            }

            /// true if all of the bytes have been written
            fn done(&self) -> bool {
                self.flushed >= *self.encoder_buffered
            }
        }

        impl Drop for BufGuard<'_> {
            fn drop(&mut self) {
                if self.flushed > 0 {
                    if self.done() {
                        *self.encoder_flushed += *self.encoder_buffered;
                        *self.encoder_buffered = 0;
                    } else {
                        self.buffer.copy_within(self.flushed.., 0);
                        *self.encoder_flushed += self.flushed;
                        *self.encoder_buffered -= self.flushed;
                    }
                }
            }
        }

        let mut guard = BufGuard::new(
            unsafe { MaybeUninit::slice_assume_init_mut(&mut self.buf[..self.buffered]) },
            &mut self.buffered,
            &mut self.flushed,
        );

        while !guard.done() {
            match self.file.write(guard.remaining()) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "failed to write the buffered data",
                    ));
                }
                Ok(n) => guard.consume(n),
                Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }

        Ok(())
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.buf.len()
    }

    #[inline]
    fn write_one(&mut self, value: u8) -> FileEncodeResult {
        // We ensure this during `FileEncoder` construction.
        debug_assert!(self.capacity() >= 1);

        let mut buffered = self.buffered;

        if std::intrinsics::unlikely(buffered >= self.capacity()) {
            self.flush()?;
            buffered = 0;
        }

        // SAFETY: The above check and `flush` ensures that there is enough
        // room to write the input to the buffer.
        unsafe {
            *MaybeUninit::slice_as_mut_ptr(&mut self.buf).add(buffered) = value;
        }

        self.buffered = buffered + 1;

        Ok(())
    }

    #[inline]
    fn write_all(&mut self, buf: &[u8]) -> FileEncodeResult {
        let capacity = self.capacity();
        let buf_len = buf.len();

        if std::intrinsics::likely(buf_len <= capacity) {
            let mut buffered = self.buffered;

            if std::intrinsics::unlikely(buf_len > capacity - buffered) {
                self.flush()?;
                buffered = 0;
            }

            // SAFETY: The above check and `flush` ensures that there is enough
            // room to write the input to the buffer.
            unsafe {
                let src = buf.as_ptr();
                let dst = MaybeUninit::slice_as_mut_ptr(&mut self.buf).add(buffered);
                ptr::copy_nonoverlapping(src, dst, buf_len);
            }

            self.buffered = buffered + buf_len;

            Ok(())
        } else {
            self.write_all_unbuffered(buf)
        }
    }

    fn write_all_unbuffered(&mut self, mut buf: &[u8]) -> FileEncodeResult {
        if self.buffered > 0 {
            self.flush()?;
        }

        // This is basically a copy of `Write::write_all` but also updates our
        // `self.flushed`. It's necessary because `Write::write_all` does not
        // return the number of bytes written when an error is encountered, and
        // without that, we cannot accurately update `self.flushed` on error.
        while !buf.is_empty() {
            match self.file.write(buf) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "failed to write whole buffer",
                    ));
                }
                Ok(n) => {
                    buf = &buf[n..];
                    self.flushed += n;
                }
                Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }

        Ok(())
    }
}

impl Drop for FileEncoder {
    fn drop(&mut self) {
        let _result = self.flush();
    }
}

macro_rules! file_encoder_write_leb128 {
    ($enc:expr, $value:expr, $int_ty:ty, $fun:ident) => {{
        const MAX_ENCODED_LEN: usize = max_leb128_len!($int_ty);

        // We ensure this during `FileEncoder` construction.
        debug_assert!($enc.capacity() >= MAX_ENCODED_LEN);

        let mut buffered = $enc.buffered;

        // This can't overflow. See assertion in `FileEncoder::with_capacity`.
        if std::intrinsics::unlikely(buffered + MAX_ENCODED_LEN > $enc.capacity()) {
            $enc.flush()?;
            buffered = 0;
        }

        // SAFETY: The above check and flush ensures that there is enough
        // room to write the encoded value to the buffer.
        let buf = unsafe {
            &mut *($enc.buf.as_mut_ptr().add(buffered) as *mut [MaybeUninit<u8>; MAX_ENCODED_LEN])
        };

        let encoded = leb128::$fun(buf, $value);
        $enc.buffered = buffered + encoded.len();

        Ok(())
    }};
}

impl serialize::Encoder for FileEncoder {
    type Error = io::Error;

    #[inline]
    fn emit_unit(&mut self) -> FileEncodeResult {
        Ok(())
    }

    #[inline]
    fn emit_usize(&mut self, v: usize) -> FileEncodeResult {
        file_encoder_write_leb128!(self, v, usize, write_usize_leb128)
    }

    #[inline]
    fn emit_u128(&mut self, v: u128) -> FileEncodeResult {
        file_encoder_write_leb128!(self, v, u128, write_u128_leb128)
    }

    #[inline]
    fn emit_u64(&mut self, v: u64) -> FileEncodeResult {
        file_encoder_write_leb128!(self, v, u64, write_u64_leb128)
    }

    #[inline]
    fn emit_u32(&mut self, v: u32) -> FileEncodeResult {
        file_encoder_write_leb128!(self, v, u32, write_u32_leb128)
    }

    #[inline]
    fn emit_u16(&mut self, v: u16) -> FileEncodeResult {
        self.write_all(&v.to_le_bytes())
    }

    #[inline]
    fn emit_u8(&mut self, v: u8) -> FileEncodeResult {
        self.write_one(v)
    }

    #[inline]
    fn emit_isize(&mut self, v: isize) -> FileEncodeResult {
        file_encoder_write_leb128!(self, v, isize, write_isize_leb128)
    }

    #[inline]
    fn emit_i128(&mut self, v: i128) -> FileEncodeResult {
        file_encoder_write_leb128!(self, v, i128, write_i128_leb128)
    }

    #[inline]
    fn emit_i64(&mut self, v: i64) -> FileEncodeResult {
        file_encoder_write_leb128!(self, v, i64, write_i64_leb128)
    }

    #[inline]
    fn emit_i32(&mut self, v: i32) -> FileEncodeResult {
        file_encoder_write_leb128!(self, v, i32, write_i32_leb128)
    }

    #[inline]
    fn emit_i16(&mut self, v: i16) -> FileEncodeResult {
        self.write_all(&v.to_le_bytes())
    }

    #[inline]
    fn emit_i8(&mut self, v: i8) -> FileEncodeResult {
        self.emit_u8(v as u8)
    }

    #[inline]
    fn emit_bool(&mut self, v: bool) -> FileEncodeResult {
        self.emit_u8(if v { 1 } else { 0 })
    }

    #[inline]
    fn emit_f64(&mut self, v: f64) -> FileEncodeResult {
        let as_u64: u64 = v.to_bits();
        self.emit_u64(as_u64)
    }

    #[inline]
    fn emit_f32(&mut self, v: f32) -> FileEncodeResult {
        let as_u32: u32 = v.to_bits();
        self.emit_u32(as_u32)
    }

    #[inline]
    fn emit_char(&mut self, v: char) -> FileEncodeResult {
        self.emit_u32(v as u32)
    }

    #[inline]
    fn emit_str(&mut self, v: &str) -> FileEncodeResult {
        self.emit_usize(v.len())?;
        self.emit_raw_bytes(v.as_bytes())?;
        self.emit_u8(STR_SENTINEL)
    }

    #[inline]
    fn emit_raw_bytes(&mut self, s: &[u8]) -> FileEncodeResult {
        self.write_all(s)
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
        Decoder { data, position }
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
    pub fn read_raw_bytes(&mut self, bytes: usize) -> &'a [u8] {
        let start = self.position;
        self.position += bytes;
        &self.data[start..self.position]
    }
}

macro_rules! read_leb128 {
    ($dec:expr, $fun:ident) => {{
        let (value, bytes_read) = leb128::$fun(&$dec.data[$dec.position..]);
        $dec.position += bytes_read;
        Ok(value)
    }};
}

impl<'a> serialize::Decoder for Decoder<'a> {
    type Error = String;

    #[inline]
    fn read_nil(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    #[inline]
    fn read_u128(&mut self) -> Result<u128, Self::Error> {
        read_leb128!(self, read_u128_leb128)
    }

    #[inline]
    fn read_u64(&mut self) -> Result<u64, Self::Error> {
        read_leb128!(self, read_u64_leb128)
    }

    #[inline]
    fn read_u32(&mut self) -> Result<u32, Self::Error> {
        read_leb128!(self, read_u32_leb128)
    }

    #[inline]
    fn read_u16(&mut self) -> Result<u16, Self::Error> {
        let bytes = [self.data[self.position], self.data[self.position + 1]];
        let value = u16::from_le_bytes(bytes);
        self.position += 2;
        Ok(value)
    }

    #[inline]
    fn read_u8(&mut self) -> Result<u8, Self::Error> {
        let value = self.data[self.position];
        self.position += 1;
        Ok(value)
    }

    #[inline]
    fn read_usize(&mut self) -> Result<usize, Self::Error> {
        read_leb128!(self, read_usize_leb128)
    }

    #[inline]
    fn read_i128(&mut self) -> Result<i128, Self::Error> {
        read_leb128!(self, read_i128_leb128)
    }

    #[inline]
    fn read_i64(&mut self) -> Result<i64, Self::Error> {
        read_leb128!(self, read_i64_leb128)
    }

    #[inline]
    fn read_i32(&mut self) -> Result<i32, Self::Error> {
        read_leb128!(self, read_i32_leb128)
    }

    #[inline]
    fn read_i16(&mut self) -> Result<i16, Self::Error> {
        let bytes = [self.data[self.position], self.data[self.position + 1]];
        let value = i16::from_le_bytes(bytes);
        self.position += 2;
        Ok(value)
    }

    #[inline]
    fn read_i8(&mut self) -> Result<i8, Self::Error> {
        let as_u8 = self.data[self.position];
        self.position += 1;
        unsafe { Ok(::std::mem::transmute(as_u8)) }
    }

    #[inline]
    fn read_isize(&mut self) -> Result<isize, Self::Error> {
        read_leb128!(self, read_isize_leb128)
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
        Ok(std::char::from_u32(bits).unwrap())
    }

    #[inline]
    fn read_str(&mut self) -> Result<Cow<'_, str>, Self::Error> {
        let len = self.read_usize()?;
        let sentinel = self.data[self.position + len];
        assert!(sentinel == STR_SENTINEL);
        let s = unsafe {
            std::str::from_utf8_unchecked(&self.data[self.position..self.position + len])
        };
        self.position += len + 1;
        Ok(Cow::Borrowed(s))
    }

    #[inline]
    fn error(&mut self, err: &str) -> Self::Error {
        err.to_string()
    }

    #[inline]
    fn read_raw_bytes_into(&mut self, s: &mut [u8]) -> Result<(), String> {
        let start = self.position;
        self.position += s.len();
        s.copy_from_slice(&self.data[start..self.position]);
        Ok(())
    }
}

// Specializations for contiguous byte sequences follow. The default implementations for slices
// encode and decode each element individually. This isn't necessary for `u8` slices when using
// opaque encoders and decoders, because each `u8` is unchanged by encoding and decoding.
// Therefore, we can use more efficient implementations that process the entire sequence at once.

// Specialize encoding byte slices. This specialization also applies to encoding `Vec<u8>`s, etc.,
// since the default implementations call `encode` on their slices internally.
impl serialize::Encodable<Encoder> for [u8] {
    fn encode(&self, e: &mut Encoder) -> EncodeResult {
        serialize::Encoder::emit_usize(e, self.len())?;
        e.emit_raw_bytes(self)
    }
}

impl serialize::Encodable<FileEncoder> for [u8] {
    fn encode(&self, e: &mut FileEncoder) -> FileEncodeResult {
        serialize::Encoder::emit_usize(e, self.len())?;
        e.emit_raw_bytes(self)
    }
}

// Specialize decoding `Vec<u8>`. This specialization also applies to decoding `Box<[u8]>`s, etc.,
// since the default implementations call `decode` to produce a `Vec<u8>` internally.
impl<'a> serialize::Decodable<Decoder<'a>> for Vec<u8> {
    fn decode(d: &mut Decoder<'a>) -> Result<Self, String> {
        let len = serialize::Decoder::read_usize(d)?;
        Ok(d.read_raw_bytes(len).to_owned())
    }
}

// An integer that will always encode to 8 bytes.
pub struct IntEncodedWithFixedSize(pub u64);

impl IntEncodedWithFixedSize {
    pub const ENCODED_SIZE: usize = 8;
}

impl serialize::Encodable<Encoder> for IntEncodedWithFixedSize {
    #[inline]
    fn encode(&self, e: &mut Encoder) -> EncodeResult {
        let _start_pos = e.position();
        e.emit_raw_bytes(&self.0.to_le_bytes())?;
        let _end_pos = e.position();
        debug_assert_eq!((_end_pos - _start_pos), IntEncodedWithFixedSize::ENCODED_SIZE);
        Ok(())
    }
}

impl serialize::Encodable<FileEncoder> for IntEncodedWithFixedSize {
    #[inline]
    fn encode(&self, e: &mut FileEncoder) -> FileEncodeResult {
        let _start_pos = e.position();
        e.emit_raw_bytes(&self.0.to_le_bytes())?;
        let _end_pos = e.position();
        debug_assert_eq!((_end_pos - _start_pos), IntEncodedWithFixedSize::ENCODED_SIZE);
        Ok(())
    }
}

impl<'a> serialize::Decodable<Decoder<'a>> for IntEncodedWithFixedSize {
    #[inline]
    fn decode(decoder: &mut Decoder<'a>) -> Result<IntEncodedWithFixedSize, String> {
        let _start_pos = decoder.position();
        let bytes = decoder.read_raw_bytes(IntEncodedWithFixedSize::ENCODED_SIZE);
        let _end_pos = decoder.position();
        debug_assert_eq!((_end_pos - _start_pos), IntEncodedWithFixedSize::ENCODED_SIZE);

        let value = u64::from_le_bytes(bytes.try_into().unwrap());
        Ok(IntEncodedWithFixedSize(value))
    }
}
