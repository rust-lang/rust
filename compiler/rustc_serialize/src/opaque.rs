use crate::leb128::{self, largest_max_leb128_len};
use crate::serialize::{Decodable, Decoder, Encodable, Encoder};
use std::fs::File;
use std::io::{self, Write};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::Range;
use std::path::Path;
use std::ptr;

// -----------------------------------------------------------------------------
// Encoder
// -----------------------------------------------------------------------------

pub struct MemEncoder {
    pub data: Vec<u8>,
}

impl MemEncoder {
    pub fn new() -> MemEncoder {
        MemEncoder { data: vec![] }
    }

    #[inline]
    pub fn position(&self) -> usize {
        self.data.len()
    }

    pub fn finish(self) -> Vec<u8> {
        self.data
    }
}

macro_rules! write_leb128 {
    ($enc:expr, $value:expr, $int_ty:ty, $fun:ident) => {{
        const MAX_ENCODED_LEN: usize = $crate::leb128::max_leb128_len::<$int_ty>();
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
    }};
}

/// A byte that [cannot occur in UTF8 sequences][utf8]. Used to mark the end of a string.
/// This way we can skip validation and still be relatively sure that deserialization
/// did not desynchronize.
///
/// [utf8]: https://en.wikipedia.org/w/index.php?title=UTF-8&oldid=1058865525#Codepage_layout
const STR_SENTINEL: u8 = 0xC1;

impl Encoder for MemEncoder {
    #[inline]
    fn emit_usize(&mut self, v: usize) {
        write_leb128!(self, v, usize, write_usize_leb128)
    }

    #[inline]
    fn emit_u128(&mut self, v: u128) {
        write_leb128!(self, v, u128, write_u128_leb128);
    }

    #[inline]
    fn emit_u64(&mut self, v: u64) {
        write_leb128!(self, v, u64, write_u64_leb128);
    }

    #[inline]
    fn emit_u32(&mut self, v: u32) {
        write_leb128!(self, v, u32, write_u32_leb128);
    }

    #[inline]
    fn emit_u16(&mut self, v: u16) {
        self.data.extend_from_slice(&v.to_le_bytes());
    }

    #[inline]
    fn emit_u8(&mut self, v: u8) {
        self.data.push(v);
    }

    #[inline]
    fn emit_isize(&mut self, v: isize) {
        write_leb128!(self, v, isize, write_isize_leb128)
    }

    #[inline]
    fn emit_i128(&mut self, v: i128) {
        write_leb128!(self, v, i128, write_i128_leb128)
    }

    #[inline]
    fn emit_i64(&mut self, v: i64) {
        write_leb128!(self, v, i64, write_i64_leb128)
    }

    #[inline]
    fn emit_i32(&mut self, v: i32) {
        write_leb128!(self, v, i32, write_i32_leb128)
    }

    #[inline]
    fn emit_i16(&mut self, v: i16) {
        self.data.extend_from_slice(&v.to_le_bytes());
    }

    #[inline]
    fn emit_i8(&mut self, v: i8) {
        self.emit_u8(v as u8);
    }

    #[inline]
    fn emit_bool(&mut self, v: bool) {
        self.emit_u8(if v { 1 } else { 0 });
    }

    #[inline]
    fn emit_char(&mut self, v: char) {
        self.emit_u32(v as u32);
    }

    #[inline]
    fn emit_str(&mut self, v: &str) {
        self.emit_usize(v.len());
        self.emit_raw_bytes(v.as_bytes());
        self.emit_u8(STR_SENTINEL);
    }

    #[inline]
    fn emit_raw_bytes(&mut self, s: &[u8]) {
        self.data.extend_from_slice(s);
    }
}

pub type FileEncodeResult = Result<usize, io::Error>;

/// `FileEncoder` encodes data to file via fixed-size buffer.
///
/// When encoding large amounts of data to a file, using `FileEncoder` may be
/// preferred over using `MemEncoder` to encode to a `Vec`, and then writing the
/// `Vec` to file, as the latter uses as much memory as there is encoded data,
/// while the former uses the fixed amount of memory allocated to the buffer.
/// `FileEncoder` also has the advantage of not needing to reallocate as data
/// is appended to it, but the disadvantage of requiring more error handling,
/// which has some runtime overhead.
pub struct FileEncoder {
    /// The input buffer. For adequate performance, we need more control over
    /// buffering than `BufWriter` offers. If `BufWriter` ever offers a raw
    /// buffer access API, we can use it, and remove `buf` and `buffered`.
    buf: Box<[MaybeUninit<u8>]>,
    buffered: usize,
    flushed: usize,
    file: File,
    // This is used to implement delayed error handling, as described in the
    // comment on `trait Encoder`.
    res: Result<(), io::Error>,
}

impl FileEncoder {
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        const DEFAULT_BUF_SIZE: usize = 8192;
        FileEncoder::with_capacity(path, DEFAULT_BUF_SIZE)
    }

    pub fn with_capacity<P: AsRef<Path>>(path: P, capacity: usize) -> io::Result<Self> {
        // Require capacity at least as large as the largest LEB128 encoding
        // here, so that we don't have to check or handle this on every write.
        assert!(capacity >= largest_max_leb128_len());

        // Require capacity small enough such that some capacity checks can be
        // done using guaranteed non-overflowing add rather than sub, which
        // shaves an instruction off those code paths (on x86 at least).
        assert!(capacity <= usize::MAX - largest_max_leb128_len());

        // Create the file for reading and writing, because some encoders do both
        // (e.g. the metadata encoder when -Zmeta-stats is enabled)
        let file = File::options().read(true).write(true).create(true).truncate(true).open(path)?;

        Ok(FileEncoder {
            buf: Box::new_uninit_slice(capacity),
            buffered: 0,
            flushed: 0,
            file,
            res: Ok(()),
        })
    }

    #[inline]
    pub fn position(&self) -> usize {
        // Tracking position this way instead of having a `self.position` field
        // means that we don't have to update the position on every write call.
        self.flushed + self.buffered
    }

    pub fn flush(&mut self) {
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

        // If we've already had an error, do nothing. It'll get reported after
        // `finish` is called.
        if self.res.is_err() {
            return;
        }

        let mut guard = BufGuard::new(
            unsafe { MaybeUninit::slice_assume_init_mut(&mut self.buf[..self.buffered]) },
            &mut self.buffered,
            &mut self.flushed,
        );

        while !guard.done() {
            match self.file.write(guard.remaining()) {
                Ok(0) => {
                    self.res = Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "failed to write the buffered data",
                    ));
                    return;
                }
                Ok(n) => guard.consume(n),
                Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => {
                    self.res = Err(e);
                    return;
                }
            }
        }
    }

    pub fn file(&self) -> &File {
        &self.file
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.buf.len()
    }

    #[inline]
    fn write_one(&mut self, value: u8) {
        // We ensure this during `FileEncoder` construction.
        debug_assert!(self.capacity() >= 1);

        let mut buffered = self.buffered;

        if std::intrinsics::unlikely(buffered >= self.capacity()) {
            self.flush();
            buffered = 0;
        }

        // SAFETY: The above check and `flush` ensures that there is enough
        // room to write the input to the buffer.
        unsafe {
            *MaybeUninit::slice_as_mut_ptr(&mut self.buf).add(buffered) = value;
        }

        self.buffered = buffered + 1;
    }

    #[inline]
    fn write_all(&mut self, buf: &[u8]) {
        let capacity = self.capacity();
        let buf_len = buf.len();

        if std::intrinsics::likely(buf_len <= capacity) {
            let mut buffered = self.buffered;

            if std::intrinsics::unlikely(buf_len > capacity - buffered) {
                self.flush();
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
        } else {
            self.write_all_unbuffered(buf);
        }
    }

    fn write_all_unbuffered(&mut self, mut buf: &[u8]) {
        // If we've already had an error, do nothing. It'll get reported after
        // `finish` is called.
        if self.res.is_err() {
            return;
        }

        if self.buffered > 0 {
            self.flush();
        }

        // This is basically a copy of `Write::write_all` but also updates our
        // `self.flushed`. It's necessary because `Write::write_all` does not
        // return the number of bytes written when an error is encountered, and
        // without that, we cannot accurately update `self.flushed` on error.
        while !buf.is_empty() {
            match self.file.write(buf) {
                Ok(0) => {
                    self.res = Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "failed to write whole buffer",
                    ));
                    return;
                }
                Ok(n) => {
                    buf = &buf[n..];
                    self.flushed += n;
                }
                Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => {
                    self.res = Err(e);
                    return;
                }
            }
        }
    }

    pub fn finish(mut self) -> Result<usize, io::Error> {
        self.flush();

        let res = std::mem::replace(&mut self.res, Ok(()));
        res.map(|()| self.position())
    }
}

impl Drop for FileEncoder {
    fn drop(&mut self) {
        // Likely to be a no-op, because `finish` should have been called and
        // it also flushes. But do it just in case.
        let _result = self.flush();
    }
}

macro_rules! file_encoder_write_leb128 {
    ($enc:expr, $value:expr, $int_ty:ty, $fun:ident) => {{
        const MAX_ENCODED_LEN: usize = $crate::leb128::max_leb128_len::<$int_ty>();

        // We ensure this during `FileEncoder` construction.
        debug_assert!($enc.capacity() >= MAX_ENCODED_LEN);

        let mut buffered = $enc.buffered;

        // This can't overflow. See assertion in `FileEncoder::with_capacity`.
        if std::intrinsics::unlikely(buffered + MAX_ENCODED_LEN > $enc.capacity()) {
            $enc.flush();
            buffered = 0;
        }

        // SAFETY: The above check and flush ensures that there is enough
        // room to write the encoded value to the buffer.
        let buf = unsafe {
            &mut *($enc.buf.as_mut_ptr().add(buffered) as *mut [MaybeUninit<u8>; MAX_ENCODED_LEN])
        };

        let encoded = leb128::$fun(buf, $value);
        $enc.buffered = buffered + encoded.len();
    }};
}

impl Encoder for FileEncoder {
    #[inline]
    fn emit_usize(&mut self, v: usize) {
        file_encoder_write_leb128!(self, v, usize, write_usize_leb128)
    }

    #[inline]
    fn emit_u128(&mut self, v: u128) {
        file_encoder_write_leb128!(self, v, u128, write_u128_leb128)
    }

    #[inline]
    fn emit_u64(&mut self, v: u64) {
        file_encoder_write_leb128!(self, v, u64, write_u64_leb128)
    }

    #[inline]
    fn emit_u32(&mut self, v: u32) {
        file_encoder_write_leb128!(self, v, u32, write_u32_leb128)
    }

    #[inline]
    fn emit_u16(&mut self, v: u16) {
        self.write_all(&v.to_le_bytes());
    }

    #[inline]
    fn emit_u8(&mut self, v: u8) {
        self.write_one(v);
    }

    #[inline]
    fn emit_isize(&mut self, v: isize) {
        file_encoder_write_leb128!(self, v, isize, write_isize_leb128)
    }

    #[inline]
    fn emit_i128(&mut self, v: i128) {
        file_encoder_write_leb128!(self, v, i128, write_i128_leb128)
    }

    #[inline]
    fn emit_i64(&mut self, v: i64) {
        file_encoder_write_leb128!(self, v, i64, write_i64_leb128)
    }

    #[inline]
    fn emit_i32(&mut self, v: i32) {
        file_encoder_write_leb128!(self, v, i32, write_i32_leb128)
    }

    #[inline]
    fn emit_i16(&mut self, v: i16) {
        self.write_all(&v.to_le_bytes());
    }

    #[inline]
    fn emit_i8(&mut self, v: i8) {
        self.emit_u8(v as u8);
    }

    #[inline]
    fn emit_bool(&mut self, v: bool) {
        self.emit_u8(if v { 1 } else { 0 });
    }

    #[inline]
    fn emit_char(&mut self, v: char) {
        self.emit_u32(v as u32);
    }

    #[inline]
    fn emit_str(&mut self, v: &str) {
        self.emit_usize(v.len());
        self.emit_raw_bytes(v.as_bytes());
        self.emit_u8(STR_SENTINEL);
    }

    #[inline]
    fn emit_raw_bytes(&mut self, s: &[u8]) {
        self.write_all(s);
    }
}

// -----------------------------------------------------------------------------
// Decoder
// -----------------------------------------------------------------------------

// Conceptually, `MemDecoder` wraps a `&[u8]` with a cursor into it that is always valid.
// This is implemented with three pointers, two which represent the original slice and a
// third that is our cursor.
// It is an invariant of this type that start <= current <= end.
// Additionally, the implementation of this type never modifies start and end.
pub struct MemDecoder<'a> {
    start: *const u8,
    current: *const u8,
    end: *const u8,
    _marker: PhantomData<&'a u8>,
}

impl<'a> MemDecoder<'a> {
    #[inline]
    pub fn new(data: &'a [u8], position: usize) -> MemDecoder<'a> {
        let Range { start, end } = data.as_ptr_range();
        MemDecoder { start, current: data[position..].as_ptr(), end, _marker: PhantomData }
    }

    #[inline]
    pub fn data(&self) -> &'a [u8] {
        // SAFETY: This recovers the original slice, only using members we never modify.
        unsafe { std::slice::from_raw_parts(self.start, self.len()) }
    }

    #[inline]
    pub fn len(&self) -> usize {
        // SAFETY: This recovers the length of the original slice, only using members we never modify.
        unsafe { self.end.sub_ptr(self.start) }
    }

    #[inline]
    pub fn remaining(&self) -> usize {
        // SAFETY: This type guarantees current <= end.
        unsafe { self.end.sub_ptr(self.current) }
    }

    #[cold]
    #[inline(never)]
    fn decoder_exhausted() -> ! {
        panic!("MemDecoder exhausted")
    }

    #[inline]
    fn read_byte(&mut self) -> u8 {
        if self.current == self.end {
            Self::decoder_exhausted();
        }
        // SAFETY: This type guarantees current <= end, and we just checked current == end.
        unsafe {
            let byte = *self.current;
            self.current = self.current.add(1);
            byte
        }
    }

    #[inline]
    fn read_array<const N: usize>(&mut self) -> [u8; N] {
        self.read_raw_bytes(N).try_into().unwrap()
    }

    // The trait method doesn't have a lifetime parameter, and we need a version of this
    // that definitely returns a slice based on the underlying storage as opposed to
    // the Decoder itself in order to implement read_str efficiently.
    #[inline]
    fn read_raw_bytes_inherent(&mut self, bytes: usize) -> &'a [u8] {
        if bytes > self.remaining() {
            Self::decoder_exhausted();
        }
        // SAFETY: We just checked if this range is in-bounds above.
        unsafe {
            let slice = std::slice::from_raw_parts(self.current, bytes);
            self.current = self.current.add(bytes);
            slice
        }
    }

    /// While we could manually expose manipulation of the decoder position,
    /// all current users of that method would need to reset the position later,
    /// incurring the bounds check of set_position twice.
    #[inline]
    pub fn with_position<F, T>(&mut self, pos: usize, func: F) -> T
    where
        F: Fn(&mut MemDecoder<'a>) -> T,
    {
        struct SetOnDrop<'a, 'guarded> {
            decoder: &'guarded mut MemDecoder<'a>,
            current: *const u8,
        }
        impl Drop for SetOnDrop<'_, '_> {
            fn drop(&mut self) {
                self.decoder.current = self.current;
            }
        }

        if pos >= self.len() {
            Self::decoder_exhausted();
        }
        let previous = self.current;
        // SAFETY: We just checked if this add is in-bounds above.
        unsafe {
            self.current = self.start.add(pos);
        }
        let guard = SetOnDrop { current: previous, decoder: self };
        func(guard.decoder)
    }
}

macro_rules! read_leb128 {
    ($dec:expr, $fun:ident) => {{ leb128::$fun($dec) }};
}

impl<'a> Decoder for MemDecoder<'a> {
    #[inline]
    fn position(&self) -> usize {
        // SAFETY: This type guarantees start <= current
        unsafe { self.current.sub_ptr(self.start) }
    }

    #[inline]
    fn read_u128(&mut self) -> u128 {
        read_leb128!(self, read_u128_leb128)
    }

    #[inline]
    fn read_u64(&mut self) -> u64 {
        read_leb128!(self, read_u64_leb128)
    }

    #[inline]
    fn read_u32(&mut self) -> u32 {
        read_leb128!(self, read_u32_leb128)
    }

    #[inline]
    fn read_u16(&mut self) -> u16 {
        u16::from_le_bytes(self.read_array())
    }

    #[inline]
    fn read_u8(&mut self) -> u8 {
        self.read_byte()
    }

    #[inline]
    fn read_usize(&mut self) -> usize {
        read_leb128!(self, read_usize_leb128)
    }

    #[inline]
    fn read_i128(&mut self) -> i128 {
        read_leb128!(self, read_i128_leb128)
    }

    #[inline]
    fn read_i64(&mut self) -> i64 {
        read_leb128!(self, read_i64_leb128)
    }

    #[inline]
    fn read_i32(&mut self) -> i32 {
        read_leb128!(self, read_i32_leb128)
    }

    #[inline]
    fn read_i16(&mut self) -> i16 {
        i16::from_le_bytes(self.read_array())
    }

    #[inline]
    fn read_i8(&mut self) -> i8 {
        self.read_byte() as i8
    }

    #[inline]
    fn read_isize(&mut self) -> isize {
        read_leb128!(self, read_isize_leb128)
    }

    #[inline]
    fn read_bool(&mut self) -> bool {
        let value = self.read_u8();
        value != 0
    }

    #[inline]
    fn read_char(&mut self) -> char {
        let bits = self.read_u32();
        std::char::from_u32(bits).unwrap()
    }

    #[inline]
    fn read_str(&mut self) -> &str {
        let len = self.read_usize();
        let bytes = self.read_raw_bytes_inherent(len + 1);
        assert!(bytes[len] == STR_SENTINEL);
        unsafe { std::str::from_utf8_unchecked(&bytes[..len]) }
    }

    #[inline]
    fn read_raw_bytes(&mut self, bytes: usize) -> &[u8] {
        self.read_raw_bytes_inherent(bytes)
    }

    #[inline]
    fn peek_byte(&self) -> u8 {
        if self.current == self.end {
            Self::decoder_exhausted();
        }
        // SAFETY: This type guarantees current is inbounds or one-past-the-end, which is end.
        // Since we just checked current == end, the current pointer must be inbounds.
        unsafe { *self.current }
    }
}

// Specializations for contiguous byte sequences follow. The default implementations for slices
// encode and decode each element individually. This isn't necessary for `u8` slices when using
// opaque encoders and decoders, because each `u8` is unchanged by encoding and decoding.
// Therefore, we can use more efficient implementations that process the entire sequence at once.

// Specialize encoding byte slices. This specialization also applies to encoding `Vec<u8>`s, etc.,
// since the default implementations call `encode` on their slices internally.
impl Encodable<MemEncoder> for [u8] {
    fn encode(&self, e: &mut MemEncoder) {
        Encoder::emit_usize(e, self.len());
        e.emit_raw_bytes(self);
    }
}

impl Encodable<FileEncoder> for [u8] {
    fn encode(&self, e: &mut FileEncoder) {
        Encoder::emit_usize(e, self.len());
        e.emit_raw_bytes(self);
    }
}

// Specialize decoding `Vec<u8>`. This specialization also applies to decoding `Box<[u8]>`s, etc.,
// since the default implementations call `decode` to produce a `Vec<u8>` internally.
impl<'a> Decodable<MemDecoder<'a>> for Vec<u8> {
    fn decode(d: &mut MemDecoder<'a>) -> Self {
        let len = Decoder::read_usize(d);
        d.read_raw_bytes(len).to_owned()
    }
}

/// An integer that will always encode to 8 bytes.
pub struct IntEncodedWithFixedSize(pub u64);

impl IntEncodedWithFixedSize {
    pub const ENCODED_SIZE: usize = 8;
}

impl Encodable<MemEncoder> for IntEncodedWithFixedSize {
    #[inline]
    fn encode(&self, e: &mut MemEncoder) {
        let _start_pos = e.position();
        e.emit_raw_bytes(&self.0.to_le_bytes());
        let _end_pos = e.position();
        debug_assert_eq!((_end_pos - _start_pos), IntEncodedWithFixedSize::ENCODED_SIZE);
    }
}

impl Encodable<FileEncoder> for IntEncodedWithFixedSize {
    #[inline]
    fn encode(&self, e: &mut FileEncoder) {
        let _start_pos = e.position();
        e.emit_raw_bytes(&self.0.to_le_bytes());
        let _end_pos = e.position();
        debug_assert_eq!((_end_pos - _start_pos), IntEncodedWithFixedSize::ENCODED_SIZE);
    }
}

impl<'a> Decodable<MemDecoder<'a>> for IntEncodedWithFixedSize {
    #[inline]
    fn decode(decoder: &mut MemDecoder<'a>) -> IntEncodedWithFixedSize {
        let _start_pos = decoder.position();
        let bytes = decoder.read_raw_bytes(IntEncodedWithFixedSize::ENCODED_SIZE);
        let value = u64::from_le_bytes(bytes.try_into().unwrap());
        let _end_pos = decoder.position();
        debug_assert_eq!((_end_pos - _start_pos), IntEncodedWithFixedSize::ENCODED_SIZE);

        IntEncodedWithFixedSize(value)
    }
}
