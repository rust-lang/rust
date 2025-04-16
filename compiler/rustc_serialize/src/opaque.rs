use std::fs::File;
use std::io::{self, Write};
use std::marker::PhantomData;
use std::ops::Range;
use std::path::{Path, PathBuf};

// This code is very hot and uses lots of arithmetic, avoid overflow checks for performance.
// See https://github.com/rust-lang/rust/pull/119440#issuecomment-1874255727
use crate::int_overflow::DebugStrictAdd;
use crate::leb128;
use crate::serialize::{Decodable, Decoder, Encodable, Encoder};

// -----------------------------------------------------------------------------
// Encoder
// -----------------------------------------------------------------------------

pub type FileEncodeResult = Result<usize, (PathBuf, io::Error)>;

pub const MAGIC_END_BYTES: &[u8] = b"rust-end-file";

/// The size of the buffer in `FileEncoder`.
const BUF_SIZE: usize = 64 * 1024;

/// `FileEncoder` encodes data to file via fixed-size buffer.
///
/// There used to be a `MemEncoder` type that encoded all the data into a
/// `Vec`. `FileEncoder` is better because its memory use is determined by the
/// size of the buffer, rather than the full length of the encoded data, and
/// because it doesn't need to reallocate memory along the way.
pub struct FileEncoder {
    // The input buffer. For adequate performance, we need to be able to write
    // directly to the unwritten region of the buffer, without calling copy_from_slice.
    // Note that our buffer is always initialized so that we can do that direct access
    // without unsafe code. Users of this type write many more than BUF_SIZE bytes, so the
    // initialization is approximately free.
    buf: Box<[u8; BUF_SIZE]>,
    buffered: usize,
    flushed: usize,
    file: File,
    // This is used to implement delayed error handling, as described in the
    // comment on `trait Encoder`.
    res: Result<(), io::Error>,
    path: PathBuf,
    #[cfg(debug_assertions)]
    finished: bool,
}

impl FileEncoder {
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        // File::create opens the file for writing only. When -Zmeta-stats is enabled, the metadata
        // encoder rewinds the file to inspect what was written. So we need to always open the file
        // for reading and writing.
        let file =
            File::options().read(true).write(true).create(true).truncate(true).open(&path)?;

        Ok(FileEncoder {
            buf: vec![0u8; BUF_SIZE].into_boxed_slice().try_into().unwrap(),
            path: path.as_ref().into(),
            buffered: 0,
            flushed: 0,
            file,
            res: Ok(()),
            #[cfg(debug_assertions)]
            finished: false,
        })
    }

    #[inline]
    pub fn position(&self) -> usize {
        // Tracking position this way instead of having a `self.position` field
        // means that we only need to update `self.buffered` on a write call,
        // as opposed to updating `self.position` and `self.buffered`.
        self.flushed.debug_strict_add(self.buffered)
    }

    #[cold]
    #[inline(never)]
    pub fn flush(&mut self) {
        #[cfg(debug_assertions)]
        {
            self.finished = false;
        }
        if self.res.is_ok() {
            self.res = self.file.write_all(&self.buf[..self.buffered]);
        }
        self.flushed += self.buffered;
        self.buffered = 0;
    }

    pub fn file(&self) -> &File {
        &self.file
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    #[inline]
    fn buffer_empty(&mut self) -> &mut [u8] {
        // SAFETY: self.buffered is inbounds as an invariant of the type
        unsafe { self.buf.get_unchecked_mut(self.buffered..) }
    }

    #[cold]
    #[inline(never)]
    fn write_all_cold_path(&mut self, buf: &[u8]) {
        self.flush();
        if let Some(dest) = self.buf.get_mut(..buf.len()) {
            dest.copy_from_slice(buf);
            self.buffered += buf.len();
        } else {
            if self.res.is_ok() {
                self.res = self.file.write_all(buf);
            }
            self.flushed += buf.len();
        }
    }

    #[inline]
    fn write_all(&mut self, buf: &[u8]) {
        #[cfg(debug_assertions)]
        {
            self.finished = false;
        }
        if let Some(dest) = self.buffer_empty().get_mut(..buf.len()) {
            dest.copy_from_slice(buf);
            self.buffered = self.buffered.debug_strict_add(buf.len());
        } else {
            self.write_all_cold_path(buf);
        }
    }

    /// Write up to `N` bytes to this encoder.
    ///
    /// This function can be used to avoid the overhead of calling memcpy for writes that
    /// have runtime-variable length, but are small and have a small fixed upper bound.
    ///
    /// This can be used to do in-place encoding as is done for leb128 (without this function
    /// we would need to write to a temporary buffer then memcpy into the encoder), and it can
    /// also be used to implement the varint scheme we use for rmeta and dep graph encoding,
    /// where we only want to encode the first few bytes of an integer. Copying in the whole
    /// integer then only advancing the encoder state for the few bytes we care about is more
    /// efficient than calling [`FileEncoder::write_all`], because variable-size copies are
    /// always lowered to `memcpy`, which has overhead and contains a lot of logic we can bypass
    /// with this function. Note that common architectures support fixed-size writes up to 8 bytes
    /// with one instruction, so while this does in some sense do wasted work, we come out ahead.
    #[inline]
    pub fn write_with<const N: usize>(&mut self, visitor: impl FnOnce(&mut [u8; N]) -> usize) {
        #[cfg(debug_assertions)]
        {
            self.finished = false;
        }
        let flush_threshold = const { BUF_SIZE.checked_sub(N).unwrap() };
        if std::intrinsics::unlikely(self.buffered > flush_threshold) {
            self.flush();
        }
        // SAFETY: We checked above that N < self.buffer_empty().len(),
        // and if isn't, flush ensures that our empty buffer is now BUF_SIZE.
        // We produce a post-mono error if N > BUF_SIZE.
        let buf = unsafe { self.buffer_empty().first_chunk_mut::<N>().unwrap_unchecked() };
        let written = visitor(buf);
        // We have to ensure that an errant visitor cannot cause self.buffered to exceed BUF_SIZE.
        if written > N {
            Self::panic_invalid_write::<N>(written);
        }
        self.buffered = self.buffered.debug_strict_add(written);
    }

    #[cold]
    #[inline(never)]
    fn panic_invalid_write<const N: usize>(written: usize) {
        panic!("FileEncoder::write_with::<{N}> cannot be used to write {written} bytes");
    }

    /// Helper for calls where [`FileEncoder::write_with`] always writes the whole array.
    #[inline]
    pub fn write_array<const N: usize>(&mut self, buf: [u8; N]) {
        self.write_with(|dest| {
            *dest = buf;
            N
        })
    }

    pub fn finish(&mut self) -> FileEncodeResult {
        self.write_all(MAGIC_END_BYTES);
        self.flush();
        #[cfg(debug_assertions)]
        {
            self.finished = true;
        }
        match std::mem::replace(&mut self.res, Ok(())) {
            Ok(()) => Ok(self.position()),
            Err(e) => Err((self.path.clone(), e)),
        }
    }
}

#[cfg(debug_assertions)]
impl Drop for FileEncoder {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            assert!(self.finished);
        }
    }
}

macro_rules! write_leb128 {
    ($this_fn:ident, $int_ty:ty, $write_leb_fn:ident) => {
        #[inline]
        fn $this_fn(&mut self, v: $int_ty) {
            self.write_with(|buf| leb128::$write_leb_fn(buf, v))
        }
    };
}

impl Encoder for FileEncoder {
    write_leb128!(emit_usize, usize, write_usize_leb128);
    write_leb128!(emit_u128, u128, write_u128_leb128);
    write_leb128!(emit_u64, u64, write_u64_leb128);
    write_leb128!(emit_u32, u32, write_u32_leb128);

    #[inline]
    fn emit_u16(&mut self, v: u16) {
        self.write_array(v.to_le_bytes());
    }

    #[inline]
    fn emit_u8(&mut self, v: u8) {
        self.write_array([v]);
    }

    write_leb128!(emit_isize, isize, write_isize_leb128);
    write_leb128!(emit_i128, i128, write_i128_leb128);
    write_leb128!(emit_i64, i64, write_i64_leb128);
    write_leb128!(emit_i32, i32, write_i32_leb128);

    #[inline]
    fn emit_i16(&mut self, v: i16) {
        self.write_array(v.to_le_bytes());
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
    pub fn new(data: &'a [u8], position: usize) -> Result<MemDecoder<'a>, ()> {
        let data = data.strip_suffix(MAGIC_END_BYTES).ok_or(())?;
        let Range { start, end } = data.as_ptr_range();
        Ok(MemDecoder { start, current: data[position..].as_ptr(), end, _marker: PhantomData })
    }

    #[inline]
    pub fn split_at(&self, position: usize) -> MemDecoder<'a> {
        assert!(position <= self.len());
        // SAFETY: We checked above that this offset is within the original slice
        let current = unsafe { self.start.add(position) };
        MemDecoder { start: self.start, current, end: self.end, _marker: PhantomData }
    }

    #[inline]
    pub fn len(&self) -> usize {
        // SAFETY: This recovers the length of the original slice, only using members we never modify.
        unsafe { self.end.offset_from_unsigned(self.start) }
    }

    #[inline]
    pub fn remaining(&self) -> usize {
        // SAFETY: This type guarantees current <= end.
        unsafe { self.end.offset_from_unsigned(self.current) }
    }

    #[cold]
    #[inline(never)]
    fn decoder_exhausted() -> ! {
        panic!("MemDecoder exhausted")
    }

    #[inline]
    pub fn read_array<const N: usize>(&mut self) -> [u8; N] {
        self.read_raw_bytes(N).try_into().unwrap()
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
    ($this_fn:ident, $int_ty:ty, $read_leb_fn:ident) => {
        #[inline]
        fn $this_fn(&mut self) -> $int_ty {
            leb128::$read_leb_fn(self)
        }
    };
}

impl<'a> Decoder for MemDecoder<'a> {
    read_leb128!(read_usize, usize, read_usize_leb128);
    read_leb128!(read_u128, u128, read_u128_leb128);
    read_leb128!(read_u64, u64, read_u64_leb128);
    read_leb128!(read_u32, u32, read_u32_leb128);

    #[inline]
    fn read_u16(&mut self) -> u16 {
        u16::from_le_bytes(self.read_array())
    }

    #[inline]
    fn read_u8(&mut self) -> u8 {
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

    read_leb128!(read_isize, isize, read_isize_leb128);
    read_leb128!(read_i128, i128, read_i128_leb128);
    read_leb128!(read_i64, i64, read_i64_leb128);
    read_leb128!(read_i32, i32, read_i32_leb128);

    #[inline]
    fn read_i16(&mut self) -> i16 {
        i16::from_le_bytes(self.read_array())
    }

    #[inline]
    fn read_raw_bytes(&mut self, bytes: usize) -> &'a [u8] {
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

    #[inline]
    fn peek_byte(&self) -> u8 {
        if self.current == self.end {
            Self::decoder_exhausted();
        }
        // SAFETY: This type guarantees current is inbounds or one-past-the-end, which is end.
        // Since we just checked current == end, the current pointer must be inbounds.
        unsafe { *self.current }
    }

    #[inline]
    fn position(&self) -> usize {
        // SAFETY: This type guarantees start <= current
        unsafe { self.current.offset_from_unsigned(self.start) }
    }
}

// Specializations for contiguous byte sequences follow. The default implementations for slices
// encode and decode each element individually. This isn't necessary for `u8` slices when using
// opaque encoders and decoders, because each `u8` is unchanged by encoding and decoding.
// Therefore, we can use more efficient implementations that process the entire sequence at once.

// Specialize encoding byte slices. This specialization also applies to encoding `Vec<u8>`s, etc.,
// since the default implementations call `encode` on their slices internally.
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

impl Encodable<FileEncoder> for IntEncodedWithFixedSize {
    #[inline]
    fn encode(&self, e: &mut FileEncoder) {
        let start_pos = e.position();
        e.write_array(self.0.to_le_bytes());
        let end_pos = e.position();
        debug_assert_eq!((end_pos - start_pos), IntEncodedWithFixedSize::ENCODED_SIZE);
    }
}

impl<'a> Decodable<MemDecoder<'a>> for IntEncodedWithFixedSize {
    #[inline]
    fn decode(decoder: &mut MemDecoder<'a>) -> IntEncodedWithFixedSize {
        let bytes = decoder.read_array::<{ IntEncodedWithFixedSize::ENCODED_SIZE }>();
        IntEncodedWithFixedSize(u64::from_le_bytes(bytes))
    }
}

#[cfg(test)]
mod tests;
