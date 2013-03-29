// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Basic input/output

*/

use result::Result;

use int;
use libc;
use libc::{c_int, c_long, c_void, size_t, ssize_t};
use libc::consts::os::posix88::*;
use os;
use cast;
use path::Path;
use ops::Drop;
use ptr;
use result;
use str;
use uint;
use vec;

#[allow(non_camel_case_types)] // not sure what to do about this
pub type fd_t = c_int;

pub mod rustrt {
    use libc;

    #[abi = "cdecl"]
    #[link_name = "rustrt"]
    pub extern {
        unsafe fn rust_get_stdin() -> *libc::FILE;
        unsafe fn rust_get_stdout() -> *libc::FILE;
        unsafe fn rust_get_stderr() -> *libc::FILE;
    }
}

// Reading

// FIXME (#2004): This is all buffered. We might need an unbuffered variant
// as well
pub enum SeekStyle { SeekSet, SeekEnd, SeekCur, }


/// The raw underlying reader trait. All readers must implement this.
pub trait Reader {
    // FIXME (#2004): Seekable really should be orthogonal.

    /// Read up to len bytes (or EOF) and put them into bytes (which
    /// must be at least len bytes long). Return number of bytes read.
    // FIXME (#2982): This should probably return an error.
    fn read(&self, bytes: &mut [u8], len: uint) -> uint;

    /// Read a single byte, returning a negative value for EOF or read error.
    fn read_byte(&self) -> int;

    /// Return whether the stream is currently at EOF position.
    fn eof(&self) -> bool;

    /// Move the current position within the stream. The second parameter
    /// determines the position that the first parameter is relative to.
    fn seek(&self, position: int, style: SeekStyle);

    /// Return the current position within the stream.
    fn tell(&self) -> uint;
}

impl Reader for @Reader {
    fn read(&self, bytes: &mut [u8], len: uint) -> uint {
        self.read(bytes, len)
    }
    fn read_byte(&self) -> int {
        self.read_byte()
    }
    fn eof(&self) -> bool {
        self.eof()
    }
    fn seek(&self, position: int, style: SeekStyle) {
        self.seek(position, style)
    }
    fn tell(&self) -> uint {
        self.tell()
    }
}

/// Generic utility functions defined on readers.
pub trait ReaderUtil {

    /// Read len bytes into a new vec.
    fn read_bytes(&self, len: uint) -> ~[u8];

    /// Read up until a specified character (which is optionally included) or EOF.
    fn read_until(&self, c: char, include: bool) -> ~str;

    /// Read up until the first '\n' char (which is not returned), or EOF.
    fn read_line(&self) -> ~str;

    /// Read n utf-8 encoded chars.
    fn read_chars(&self, n: uint) -> ~[char];

    /// Read a single utf-8 encoded char.
    fn read_char(&self) -> char;

    /// Read up until the first null byte (which is not returned), or EOF.
    fn read_c_str(&self) -> ~str;

    /// Read all the data remaining in the stream in one go.
    fn read_whole_stream(&self) -> ~[u8];

    /// Iterate over every byte until the iterator breaks or EOF.
    fn each_byte(&self, it: &fn(int) -> bool);

    /// Iterate over every char until the iterator breaks or EOF.
    fn each_char(&self, it: &fn(char) -> bool);

    /// Iterate over every line until the iterator breaks or EOF.
    fn each_line(&self, it: &fn(&str) -> bool);

    /// Read all the lines of the file into a vector.
    fn read_lines(&self) -> ~[~str];

    /// Read n (between 1 and 8) little-endian unsigned integer bytes.
    fn read_le_uint_n(&self, nbytes: uint) -> u64;

    /// Read n (between 1 and 8) little-endian signed integer bytes.
    fn read_le_int_n(&self, nbytes: uint) -> i64;

    /// Read n (between 1 and 8) big-endian unsigned integer bytes.
    fn read_be_uint_n(&self, nbytes: uint) -> u64;

    /// Read n (between 1 and 8) big-endian signed integer bytes.
    fn read_be_int_n(&self, nbytes: uint) -> i64;

    /// Read a little-endian uint (number of bytes depends on system).
    fn read_le_uint(&self) -> uint;

    /// Read a little-endian int (number of bytes depends on system).
    fn read_le_int(&self) -> int;

    /// Read a big-endian uint (number of bytes depends on system).
    fn read_be_uint(&self) -> uint;

    /// Read a big-endian int (number of bytes depends on system).
    fn read_be_int(&self) -> int;

    /// Read a big-endian u64 (8 bytes).
    fn read_be_u64(&self) -> u64;

    /// Read a big-endian u32 (4 bytes).
    fn read_be_u32(&self) -> u32;

    /// Read a big-endian u16 (2 bytes).
    fn read_be_u16(&self) -> u16;

    /// Read a big-endian i64 (8 bytes).
    fn read_be_i64(&self) -> i64;

    /// Read a big-endian i32 (4 bytes).
    fn read_be_i32(&self) -> i32;

    /// Read a big-endian i16 (2 bytes).
    fn read_be_i16(&self) -> i16;

    /// Read a big-endian IEEE754 double-precision floating-point (8 bytes).
    fn read_be_f64(&self) -> f64;

    /// Read a big-endian IEEE754 single-precision floating-point (4 bytes).
    fn read_be_f32(&self) -> f32;

    /// Read a little-endian u64 (8 bytes).
    fn read_le_u64(&self) -> u64;

    /// Read a little-endian u32 (4 bytes).
    fn read_le_u32(&self) -> u32;

    /// Read a little-endian u16 (2 bytes).
    fn read_le_u16(&self) -> u16;

    /// Read a litle-endian i64 (8 bytes).
    fn read_le_i64(&self) -> i64;

    /// Read a litle-endian i32 (4 bytes).
    fn read_le_i32(&self) -> i32;

    /// Read a litle-endian i16 (2 bytes).
    fn read_le_i16(&self) -> i16;

    /// Read a litten-endian IEEE754 double-precision floating-point
    /// (8 bytes).
    fn read_le_f64(&self) -> f64;

    /// Read a litten-endian IEEE754 single-precision floating-point
    /// (4 bytes).
    fn read_le_f32(&self) -> f32;

    /// Read a u8 (1 byte).
    fn read_u8(&self) -> u8;

    /// Read a i8 (1 byte).
    fn read_i8(&self) -> i8;
}

impl<T:Reader> ReaderUtil for T {

    fn read_bytes(&self,len: uint) -> ~[u8] {
        let mut bytes = vec::with_capacity(len);
        unsafe { vec::raw::set_len(&mut bytes, len); }

        let count = self.read(bytes, len);

        unsafe { vec::raw::set_len(&mut bytes, count); }
        bytes
    }

    fn read_until(&self, c: char, include: bool) -> ~str {
        let mut bytes = ~[];
        loop {
            let ch = self.read_byte();
            if ch == -1 || ch == c as int {
                if include && ch == c as int {
                    bytes.push(ch as u8);
                }
                break;
            }
            bytes.push(ch as u8);
        }
        str::from_bytes(bytes)
    }

    fn read_line(&self) -> ~str {
        self.read_until('\n', false)
    }

    fn read_chars(&self, n: uint) -> ~[char] {
        // returns the (consumed offset, n_req), appends characters to &chars
        fn chars_from_bytes<T:Reader>(bytes: &~[u8], chars: &mut ~[char])
            -> (uint, uint) {
            let mut i = 0;
            let bytes_len = bytes.len();
            while i < bytes_len {
                let b0 = bytes[i];
                let w = str::utf8_char_width(b0);
                let end = i + w;
                i += 1;
                assert!((w > 0));
                if w == 1 {
                    chars.push(b0 as char);
                    loop;
                }
                // can't satisfy this char with the existing data
                if end > bytes_len {
                    return (i - 1, end - bytes_len);
                }
                let mut val = 0;
                while i < end {
                    let next = bytes[i] as int;
                    i += 1;
                    assert!((next > -1));
                    assert!((next & 192 == 128));
                    val <<= 6;
                    val += (next & 63) as uint;
                }
                // See str::char_at
                val += ((b0 << ((w + 1) as u8)) as uint)
                    << (w - 1) * 6 - w - 1u;
                chars.push(val as char);
            }
            return (i, 0);
        }
        let mut bytes = ~[];
        let mut chars = ~[];
        // might need more bytes, but reading n will never over-read
        let mut nbread = n;
        while nbread > 0 {
            let data = self.read_bytes(nbread);
            if data.is_empty() {
                // eof - FIXME (#2004): should we do something if
                // we're split in a unicode char?
                break;
            }
            bytes.push_all(data);
            let (offset, nbreq) = chars_from_bytes::<T>(&bytes, &mut chars);
            let ncreq = n - chars.len();
            // again we either know we need a certain number of bytes
            // to complete a character, or we make sure we don't
            // over-read by reading 1-byte per char needed
            nbread = if ncreq > nbreq { ncreq } else { nbreq };
            if nbread > 0 {
                bytes = vec::slice(bytes, offset, bytes.len()).to_vec();
            }
        }
        chars
    }

    fn read_char(&self) -> char {
        let c = self.read_chars(1);
        if vec::len(c) == 0 {
            return -1 as char; // FIXME will this stay valid? // #2004
        }
        assert!((vec::len(c) == 1));
        return c[0];
    }

    fn read_c_str(&self) -> ~str {
        self.read_until(0 as char, false)
    }

    fn read_whole_stream(&self) -> ~[u8] {
        let mut bytes: ~[u8] = ~[];
        while !self.eof() { bytes.push_all(self.read_bytes(2048u)); }
        bytes
    }

    fn each_byte(&self, it: &fn(int) -> bool) {
        while !self.eof() {
            if !it(self.read_byte()) { break; }
        }
    }

    fn each_char(&self, it: &fn(char) -> bool) {
        while !self.eof() {
            if !it(self.read_char()) { break; }
        }
    }

    fn each_line(&self, it: &fn(s: &str) -> bool) {
        while !self.eof() {
            // include the \n, so that we can distinguish an entirely empty
            // line read after "...\n", and the trailing empty line in
            // "...\n\n".
            let mut line = self.read_until('\n', true);

            // blank line at the end of the reader is ignored
            if self.eof() && line.is_empty() { break; }

            // trim the \n, so that each_line is consistent with read_line
            let n = str::len(line);
            if line[n-1] == '\n' as u8 {
                unsafe { str::raw::set_len(&mut line, n-1); }
            }

            if !it(line) { break; }
        }
    }

    fn read_lines(&self) -> ~[~str] {
        do vec::build |push| {
            for self.each_line |line| {
                push(str::from_slice(line));
            }
        }
    }

    // FIXME int reading methods need to deal with eof - issue #2004

    fn read_le_uint_n(&self, nbytes: uint) -> u64 {
        assert!(nbytes > 0 && nbytes <= 8);

        let mut val = 0u64, pos = 0, i = nbytes;
        while i > 0 {
            val += (self.read_u8() as u64) << pos;
            pos += 8;
            i -= 1;
        }
        val
    }

    fn read_le_int_n(&self, nbytes: uint) -> i64 {
        extend_sign(self.read_le_uint_n(nbytes), nbytes)
    }

    fn read_be_uint_n(&self, nbytes: uint) -> u64 {
        assert!(nbytes > 0 && nbytes <= 8);

        let mut val = 0u64, i = nbytes;
        while i > 0 {
            i -= 1;
            val += (self.read_u8() as u64) << i * 8;
        }
        val
    }

    fn read_be_int_n(&self, nbytes: uint) -> i64 {
        extend_sign(self.read_be_uint_n(nbytes), nbytes)
    }

    fn read_le_uint(&self) -> uint {
        self.read_le_uint_n(uint::bytes) as uint
    }

    fn read_le_int(&self) -> int {
        self.read_le_int_n(int::bytes) as int
    }

    fn read_be_uint(&self) -> uint {
        self.read_be_uint_n(uint::bytes) as uint
    }

    fn read_be_int(&self) -> int {
        self.read_be_int_n(int::bytes) as int
    }

    fn read_be_u64(&self) -> u64 {
        self.read_be_uint_n(8) as u64
    }

    fn read_be_u32(&self) -> u32 {
        self.read_be_uint_n(4) as u32
    }

    fn read_be_u16(&self) -> u16 {
        self.read_be_uint_n(2) as u16
    }

    fn read_be_i64(&self) -> i64 {
        self.read_be_int_n(8) as i64
    }

    fn read_be_i32(&self) -> i32 {
        self.read_be_int_n(4) as i32
    }

    fn read_be_i16(&self) -> i16 {
        self.read_be_int_n(2) as i16
    }

    fn read_be_f64(&self) -> f64 {
        unsafe {
            cast::transmute::<u64, f64>(self.read_be_u64())
        }
    }

    fn read_be_f32(&self) -> f32 {
        unsafe {
            cast::transmute::<u32, f32>(self.read_be_u32())
        }
    }

    fn read_le_u64(&self) -> u64 {
        self.read_le_uint_n(8) as u64
    }

    fn read_le_u32(&self) -> u32 {
        self.read_le_uint_n(4) as u32
    }

    fn read_le_u16(&self) -> u16 {
        self.read_le_uint_n(2) as u16
    }

    fn read_le_i64(&self) -> i64 {
        self.read_le_int_n(8) as i64
    }

    fn read_le_i32(&self) -> i32 {
        self.read_le_int_n(4) as i32
    }

    fn read_le_i16(&self) -> i16 {
        self.read_le_int_n(2) as i16
    }

    fn read_le_f64(&self) -> f64 {
        unsafe {
            cast::transmute::<u64, f64>(self.read_le_u64())
        }
    }

    fn read_le_f32(&self) -> f32 {
        unsafe {
            cast::transmute::<u32, f32>(self.read_le_u32())
        }
    }

    fn read_u8(&self) -> u8 {
        self.read_byte() as u8
    }

    fn read_i8(&self) -> i8 {
        self.read_byte() as i8
    }
}

fn extend_sign(val: u64, nbytes: uint) -> i64 {
    let shift = (8 - nbytes) * 8;
    (val << shift) as i64 >> shift
}

// Reader implementations

fn convert_whence(whence: SeekStyle) -> i32 {
    return match whence {
      SeekSet => 0i32,
      SeekCur => 1i32,
      SeekEnd => 2i32
    };
}

impl Reader for *libc::FILE {
    fn read(&self, bytes: &mut [u8], len: uint) -> uint {
        unsafe {
            do vec::as_mut_buf(bytes) |buf_p, buf_len| {
                assert!(buf_len >= len);

                let count = libc::fread(buf_p as *mut c_void, 1u as size_t,
                                        len as size_t, *self);

                count as uint
            }
        }
    }
    fn read_byte(&self) -> int {
        unsafe {
            libc::fgetc(*self) as int
        }
    }
    fn eof(&self) -> bool {
        unsafe {
            return libc::feof(*self) != 0 as c_int;
        }
    }
    fn seek(&self, offset: int, whence: SeekStyle) {
        unsafe {
            assert!(libc::fseek(*self,
                                     offset as c_long,
                                     convert_whence(whence)) == 0 as c_int);
        }
    }
    fn tell(&self) -> uint {
        unsafe {
            return libc::ftell(*self) as uint;
        }
    }
}

struct Wrapper<T, C> {
    base: T,
    cleanup: C,
}

// A forwarding impl of reader that also holds on to a resource for the
// duration of its lifetime.
// FIXME there really should be a better way to do this // #2004
impl<R:Reader,C> Reader for Wrapper<R, C> {
    fn read(&self, bytes: &mut [u8], len: uint) -> uint {
        self.base.read(bytes, len)
    }
    fn read_byte(&self) -> int { self.base.read_byte() }
    fn eof(&self) -> bool { self.base.eof() }
    fn seek(&self, off: int, whence: SeekStyle) {
        self.base.seek(off, whence)
    }
    fn tell(&self) -> uint { self.base.tell() }
}

pub struct FILERes {
    f: *libc::FILE,
}

impl Drop for FILERes {
    fn finalize(&self) {
        unsafe {
            libc::fclose(self.f);
        }
    }
}

pub fn FILERes(f: *libc::FILE) -> FILERes {
    FILERes {
        f: f
    }
}

pub fn FILE_reader(f: *libc::FILE, cleanup: bool) -> @Reader {
    if cleanup {
        @Wrapper { base: f, cleanup: FILERes(f) } as @Reader
    } else {
        @f as @Reader
    }
}

// FIXME (#2004): this should either be an trait-less impl, a set of
// top-level functions that take a reader, or a set of default methods on
// reader (which can then be called reader)

pub fn stdin() -> @Reader {
    unsafe {
        @rustrt::rust_get_stdin() as @Reader
    }
}

pub fn file_reader(path: &Path) -> Result<@Reader, ~str> {
    unsafe {
        let f = os::as_c_charp(path.to_str(), |pathbuf| {
            os::as_c_charp("r", |modebuf|
                libc::fopen(pathbuf, modebuf)
            )
        });
        return if f as uint == 0u { result::Err(~"error opening "
                                                + path.to_str()) }
        else {
            result::Ok(FILE_reader(f, true))
        }
    }
}


// Byte readers
pub struct BytesReader<'self> {
    bytes: &'self [u8],
    mut pos: uint
}

impl<'self> Reader for BytesReader<'self> {
    fn read(&self, bytes: &mut [u8], len: uint) -> uint {
        let count = uint::min(len, self.bytes.len() - self.pos);

        let view = vec::slice(self.bytes, self.pos, self.bytes.len());
        vec::bytes::copy_memory(bytes, view, count);

        self.pos += count;

        count
    }
    fn read_byte(&self) -> int {
        if self.pos == self.bytes.len() { return -1; }
        let b = self.bytes[self.pos];
        self.pos += 1u;
        return b as int;
    }
    fn eof(&self) -> bool { self.pos == self.bytes.len() }
    fn seek(&self, offset: int, whence: SeekStyle) {
        let pos = self.pos;
        self.pos = seek_in_buf(offset, pos, self.bytes.len(), whence);
    }
    fn tell(&self) -> uint { self.pos }
}

pub fn with_bytes_reader<t>(bytes: &[u8], f: &fn(@Reader) -> t) -> t {
    f(@BytesReader { bytes: bytes, pos: 0u } as @Reader)
}

pub fn with_str_reader<T>(s: &str, f: &fn(@Reader) -> T) -> T {
    str::byte_slice(s, |bytes| with_bytes_reader(bytes, f))
}

// Writing
pub enum FileFlag { Append, Create, Truncate, NoFlag, }

// What type of writer are we?
#[deriving(Eq)]
pub enum WriterType { Screen, File }

// FIXME (#2004): Seekable really should be orthogonal.
// FIXME (#2004): eventually u64
/// The raw underlying writer trait. All writers must implement this.
pub trait Writer {

    /// Write all of the given bytes.
    fn write(&self, v: &const [u8]);

    /// Move the current position within the stream. The second parameter
    /// determines the position that the first parameter is relative to.
    fn seek(&self, int, SeekStyle);

    /// Return the current position within the stream.
    fn tell(&self) -> uint;

    /// Flush the output buffer for this stream (if there is one).
    fn flush(&self) -> int;

    /// Determine if this Writer is writing to a file or not.
    fn get_type(&self) -> WriterType;
}

impl Writer for @Writer {
    fn write(&self, v: &const [u8]) { self.write(v) }
    fn seek(&self, a: int, b: SeekStyle) { self.seek(a, b) }
    fn tell(&self) -> uint { self.tell() }
    fn flush(&self) -> int { self.flush() }
    fn get_type(&self) -> WriterType { self.get_type() }
}

impl<W:Writer,C> Writer for Wrapper<W, C> {
    fn write(&self, bs: &const [u8]) { self.base.write(bs); }
    fn seek(&self, off: int, style: SeekStyle) { self.base.seek(off, style); }
    fn tell(&self) -> uint { self.base.tell() }
    fn flush(&self) -> int { self.base.flush() }
    fn get_type(&self) -> WriterType { File }
}

impl Writer for *libc::FILE {
    fn write(&self, v: &const [u8]) {
        unsafe {
            do vec::as_const_buf(v) |vbuf, len| {
                let nout = libc::fwrite(vbuf as *c_void,
                                        1,
                                        len as size_t,
                                        *self);
                if nout != len as size_t {
                    error!("error writing buffer");
                    error!("%s", os::last_os_error());
                    fail!();
                }
            }
        }
    }
    fn seek(&self, offset: int, whence: SeekStyle) {
        unsafe {
            assert!(libc::fseek(*self,
                                     offset as c_long,
                                     convert_whence(whence)) == 0 as c_int);
        }
    }
    fn tell(&self) -> uint {
        unsafe {
            libc::ftell(*self) as uint
        }
    }
    fn flush(&self) -> int {
        unsafe {
            libc::fflush(*self) as int
        }
    }
    fn get_type(&self) -> WriterType {
        unsafe {
            let fd = libc::fileno(*self);
            if libc::isatty(fd) == 0 { File   }
            else                     { Screen }
        }
    }
}

pub fn FILE_writer(f: *libc::FILE, cleanup: bool) -> @Writer {
    if cleanup {
        @Wrapper { base: f, cleanup: FILERes(f) } as @Writer
    } else {
        @f as @Writer
    }
}

impl Writer for fd_t {
    fn write(&self, v: &const [u8]) {
        unsafe {
            let mut count = 0u;
            do vec::as_const_buf(v) |vbuf, len| {
                while count < len {
                    let vb = ptr::const_offset(vbuf, count) as *c_void;
                    let nout = libc::write(*self, vb, len as size_t);
                    if nout < 0 as ssize_t {
                        error!("error writing buffer");
                        error!("%s", os::last_os_error());
                        fail!();
                    }
                    count += nout as uint;
                }
            }
        }
    }
    fn seek(&self, _offset: int, _whence: SeekStyle) {
        error!("need 64-bit foreign calls for seek, sorry");
        fail!();
    }
    fn tell(&self) -> uint {
        error!("need 64-bit foreign calls for tell, sorry");
        fail!();
    }
    fn flush(&self) -> int { 0 }
    fn get_type(&self) -> WriterType {
        unsafe {
            if libc::isatty(*self) == 0 { File } else { Screen }
        }
    }
}

pub struct FdRes {
    fd: fd_t,
}

impl Drop for FdRes {
    fn finalize(&self) {
        unsafe {
            libc::close(self.fd);
        }
    }
}

pub fn FdRes(fd: fd_t) -> FdRes {
    FdRes {
        fd: fd
    }
}

pub fn fd_writer(fd: fd_t, cleanup: bool) -> @Writer {
    if cleanup {
        @Wrapper { base: fd, cleanup: FdRes(fd) } as @Writer
    } else {
        @fd as @Writer
    }
}


pub fn mk_file_writer(path: &Path, flags: &[FileFlag])
                   -> Result<@Writer, ~str> {
    #[cfg(windows)]
    fn wb() -> c_int {
      (O_WRONLY | libc::consts::os::extra::O_BINARY) as c_int
    }

    #[cfg(unix)]
    fn wb() -> c_int { O_WRONLY as c_int }

    let mut fflags: c_int = wb();
    for vec::each(flags) |f| {
        match *f {
          Append => fflags |= O_APPEND as c_int,
          Create => fflags |= O_CREAT as c_int,
          Truncate => fflags |= O_TRUNC as c_int,
          NoFlag => ()
        }
    }
    let fd = unsafe {
        do os::as_c_charp(path.to_str()) |pathbuf| {
            libc::open(pathbuf, fflags,
                       (S_IRUSR | S_IWUSR) as c_int)
        }
    };
    if fd < (0 as c_int) {
        result::Err(fmt!("error opening %s: %s", path.to_str(),
                         os::last_os_error()))
    } else {
        result::Ok(fd_writer(fd, true))
    }
}

pub fn u64_to_le_bytes<T>(n: u64, size: uint,
                          f: &fn(v: &[u8]) -> T) -> T {
    assert!(size <= 8u);
    match size {
      1u => f(&[n as u8]),
      2u => f(&[n as u8,
              (n >> 8) as u8]),
      4u => f(&[n as u8,
              (n >> 8) as u8,
              (n >> 16) as u8,
              (n >> 24) as u8]),
      8u => f(&[n as u8,
              (n >> 8) as u8,
              (n >> 16) as u8,
              (n >> 24) as u8,
              (n >> 32) as u8,
              (n >> 40) as u8,
              (n >> 48) as u8,
              (n >> 56) as u8]),
      _ => {

        let mut bytes: ~[u8] = ~[], i = size, n = n;
        while i > 0u {
            bytes.push((n & 255_u64) as u8);
            n >>= 8_u64;
            i -= 1u;
        }
        f(bytes)
      }
    }
}

pub fn u64_to_be_bytes<T>(n: u64, size: uint,
                           f: &fn(v: &[u8]) -> T) -> T {
    assert!(size <= 8u);
    match size {
      1u => f(&[n as u8]),
      2u => f(&[(n >> 8) as u8,
              n as u8]),
      4u => f(&[(n >> 24) as u8,
              (n >> 16) as u8,
              (n >> 8) as u8,
              n as u8]),
      8u => f(&[(n >> 56) as u8,
              (n >> 48) as u8,
              (n >> 40) as u8,
              (n >> 32) as u8,
              (n >> 24) as u8,
              (n >> 16) as u8,
              (n >> 8) as u8,
              n as u8]),
      _ => {
        let mut bytes: ~[u8] = ~[];
        let mut i = size;
        while i > 0u {
            let shift = ((i - 1u) * 8u) as u64;
            bytes.push((n >> shift) as u8);
            i -= 1u;
        }
        f(bytes)
      }
    }
}

pub fn u64_from_be_bytes(data: &const [u8],
                         start: uint,
                         size: uint)
                      -> u64 {
    let mut sz = size;
    assert!((sz <= 8u));
    let mut val = 0_u64;
    let mut pos = start;
    while sz > 0u {
        sz -= 1u;
        val += (data[pos] as u64) << ((sz * 8u) as u64);
        pos += 1u;
    }
    return val;
}

// FIXME: #3048 combine trait+impl (or just move these to
// default methods on writer)
/// Generic utility functions defined on writers.
pub trait WriterUtil {

    /// Write a single utf-8 encoded char.
    fn write_char(&self, ch: char);

    /// Write every char in the given str, encoded as utf-8.
    fn write_str(&self, s: &str);

    /// Write the given str, as utf-8, followed by '\n'.
    fn write_line(&self, s: &str);

    /// Write the result of passing n through `int::to_str_bytes`.
    fn write_int(&self, n: int);

    /// Write the result of passing n through `uint::to_str_bytes`.
    fn write_uint(&self, n: uint);

    /// Write a little-endian uint (number of bytes depends on system).
    fn write_le_uint(&self, n: uint);

    /// Write a little-endian int (number of bytes depends on system).
    fn write_le_int(&self, n: int);

    /// Write a big-endian uint (number of bytes depends on system).
    fn write_be_uint(&self, n: uint);

    /// Write a big-endian int (number of bytes depends on system).
    fn write_be_int(&self, n: int);

    /// Write a big-endian u64 (8 bytes).
    fn write_be_u64(&self, n: u64);

    /// Write a big-endian u32 (4 bytes).
    fn write_be_u32(&self, n: u32);

    /// Write a big-endian u16 (2 bytes).
    fn write_be_u16(&self, n: u16);

    /// Write a big-endian i64 (8 bytes).
    fn write_be_i64(&self, n: i64);

    /// Write a big-endian i32 (4 bytes).
    fn write_be_i32(&self, n: i32);

    /// Write a big-endian i16 (2 bytes).
    fn write_be_i16(&self, n: i16);

    /// Write a big-endian IEEE754 double-precision floating-point (8 bytes).
    fn write_be_f64(&self, f: f64);

    /// Write a big-endian IEEE754 single-precision floating-point (4 bytes).
    fn write_be_f32(&self, f: f32);

    /// Write a little-endian u64 (8 bytes).
    fn write_le_u64(&self, n: u64);

    /// Write a little-endian u32 (4 bytes).
    fn write_le_u32(&self, n: u32);

    /// Write a little-endian u16 (2 bytes).
    fn write_le_u16(&self, n: u16);

    /// Write a little-endian i64 (8 bytes).
    fn write_le_i64(&self, n: i64);

    /// Write a little-endian i32 (4 bytes).
    fn write_le_i32(&self, n: i32);

    /// Write a little-endian i16 (2 bytes).
    fn write_le_i16(&self, n: i16);

    /// Write a little-endian IEEE754 double-precision floating-point
    /// (8 bytes).
    fn write_le_f64(&self, f: f64);

    /// Write a litten-endian IEEE754 single-precision floating-point
    /// (4 bytes).
    fn write_le_f32(&self, f: f32);

    /// Write a u8 (1 byte).
    fn write_u8(&self, n: u8);

    /// Write a i8 (1 byte).
    fn write_i8(&self, n: i8);
}

impl<T:Writer> WriterUtil for T {
    fn write_char(&self, ch: char) {
        if (ch as uint) < 128u {
            self.write(&[ch as u8]);
        } else {
            self.write_str(str::from_char(ch));
        }
    }
    fn write_str(&self, s: &str) { str::byte_slice(s, |v| self.write(v)) }
    fn write_line(&self, s: &str) {
        self.write_str(s);
        self.write_str(&"\n");
    }
    fn write_int(&self, n: int) {
        int::to_str_bytes(n, 10u, |bytes| self.write(bytes))
    }
    fn write_uint(&self, n: uint) {
        uint::to_str_bytes(n, 10u, |bytes| self.write(bytes))
    }
    fn write_le_uint(&self, n: uint) {
        u64_to_le_bytes(n as u64, uint::bytes, |v| self.write(v))
    }
    fn write_le_int(&self, n: int) {
        u64_to_le_bytes(n as u64, int::bytes, |v| self.write(v))
    }
    fn write_be_uint(&self, n: uint) {
        u64_to_be_bytes(n as u64, uint::bytes, |v| self.write(v))
    }
    fn write_be_int(&self, n: int) {
        u64_to_be_bytes(n as u64, int::bytes, |v| self.write(v))
    }
    fn write_be_u64(&self, n: u64) {
        u64_to_be_bytes(n, 8u, |v| self.write(v))
    }
    fn write_be_u32(&self, n: u32) {
        u64_to_be_bytes(n as u64, 4u, |v| self.write(v))
    }
    fn write_be_u16(&self, n: u16) {
        u64_to_be_bytes(n as u64, 2u, |v| self.write(v))
    }
    fn write_be_i64(&self, n: i64) {
        u64_to_be_bytes(n as u64, 8u, |v| self.write(v))
    }
    fn write_be_i32(&self, n: i32) {
        u64_to_be_bytes(n as u64, 4u, |v| self.write(v))
    }
    fn write_be_i16(&self, n: i16) {
        u64_to_be_bytes(n as u64, 2u, |v| self.write(v))
    }
    fn write_be_f64(&self, f:f64) {
        unsafe {
            self.write_be_u64(cast::transmute(f))
        }
    }
    fn write_be_f32(&self, f:f32) {
        unsafe {
            self.write_be_u32(cast::transmute(f))
        }
    }
    fn write_le_u64(&self, n: u64) {
        u64_to_le_bytes(n, 8u, |v| self.write(v))
    }
    fn write_le_u32(&self, n: u32) {
        u64_to_le_bytes(n as u64, 4u, |v| self.write(v))
    }
    fn write_le_u16(&self, n: u16) {
        u64_to_le_bytes(n as u64, 2u, |v| self.write(v))
    }
    fn write_le_i64(&self, n: i64) {
        u64_to_le_bytes(n as u64, 8u, |v| self.write(v))
    }
    fn write_le_i32(&self, n: i32) {
        u64_to_le_bytes(n as u64, 4u, |v| self.write(v))
    }
    fn write_le_i16(&self, n: i16) {
        u64_to_le_bytes(n as u64, 2u, |v| self.write(v))
    }
    fn write_le_f64(&self, f:f64) {
        unsafe {
            self.write_le_u64(cast::transmute(f))
        }
    }
    fn write_le_f32(&self, f:f32) {
        unsafe {
            self.write_le_u32(cast::transmute(f))
        }
    }

    fn write_u8(&self, n: u8) { self.write([n]) }
    fn write_i8(&self, n: i8) { self.write([n as u8]) }

}

#[allow(non_implicitly_copyable_typarams)]
pub fn file_writer(path: &Path, flags: &[FileFlag]) -> Result<@Writer, ~str> {
    mk_file_writer(path, flags).chain(|w| result::Ok(w))
}


// FIXME: fileflags // #2004
pub fn buffered_file_writer(path: &Path) -> Result<@Writer, ~str> {
    unsafe {
        let f = do os::as_c_charp(path.to_str()) |pathbuf| {
            do os::as_c_charp("w") |modebuf| {
                libc::fopen(pathbuf, modebuf)
            }
        };
        return if f as uint == 0u {
            result::Err(~"error opening " + path.to_str())
        } else {
            result::Ok(FILE_writer(f, true))
        }
    }
}

// FIXME (#2004) it would be great if this could be a const
// FIXME (#2004) why are these different from the way stdin() is
// implemented?
pub fn stdout() -> @Writer { fd_writer(libc::STDOUT_FILENO as c_int, false) }
pub fn stderr() -> @Writer { fd_writer(libc::STDERR_FILENO as c_int, false) }

pub fn print(s: &str) { stdout().write_str(s); }
pub fn println(s: &str) { stdout().write_line(s); }

pub struct BytesWriter {
    mut bytes: ~[u8],
    mut pos: uint,
}

impl Writer for BytesWriter {
    fn write(&self, v: &const [u8]) {
        let v_len = v.len();
        let bytes_len = vec::uniq_len(&const self.bytes);

        let count = uint::max(bytes_len, self.pos + v_len);
        vec::reserve(&mut self.bytes, count);

        unsafe {
            vec::raw::set_len(&mut self.bytes, count);
            let view = vec::mut_slice(self.bytes, self.pos, count);
            vec::bytes::copy_memory(view, v, v_len);
        }

        self.pos += v_len;
    }
    fn seek(&self, offset: int, whence: SeekStyle) {
        let pos = self.pos;
        let len = vec::uniq_len(&const self.bytes);
        self.pos = seek_in_buf(offset, pos, len, whence);
    }
    fn tell(&self) -> uint { self.pos }
    fn flush(&self) -> int { 0 }
    fn get_type(&self) -> WriterType { File }
}

pub fn BytesWriter() -> BytesWriter {
    BytesWriter { bytes: ~[], mut pos: 0u }
}

pub fn with_bytes_writer(f: &fn(@Writer)) -> ~[u8] {
    let wr = @BytesWriter();
    f(wr as @Writer);
    let @BytesWriter{bytes, _} = wr;
    return bytes;
}

pub fn with_str_writer(f: &fn(@Writer)) -> ~str {
    let mut v = with_bytes_writer(f);

    // FIXME (#3758): This should not be needed.
    unsafe {
        // Make sure the vector has a trailing null and is proper utf8.
        v.push(0);
    }
    assert!(str::is_utf8(v));

    unsafe { ::cast::transmute(v) }
}

// Utility functions
pub fn seek_in_buf(offset: int, pos: uint, len: uint, whence: SeekStyle) ->
   uint {
    let mut bpos = pos as int;
    let blen = len as int;
    match whence {
      SeekSet => bpos = offset,
      SeekCur => bpos += offset,
      SeekEnd => bpos = blen + offset
    }
    if bpos < 0 { bpos = 0; } else if bpos > blen { bpos = blen; }
    return bpos as uint;
}

#[allow(non_implicitly_copyable_typarams)]
pub fn read_whole_file_str(file: &Path) -> Result<~str, ~str> {
    result::chain(read_whole_file(file), |bytes| {
        if str::is_utf8(bytes) {
            result::Ok(str::from_bytes(bytes))
       } else {
           result::Err(file.to_str() + ~" is not UTF-8")
       }
    })
}

// FIXME (#2004): implement this in a low-level way. Going through the
// abstractions is pointless.
#[allow(non_implicitly_copyable_typarams)]
pub fn read_whole_file(file: &Path) -> Result<~[u8], ~str> {
    result::chain(file_reader(file), |rdr| {
        result::Ok(rdr.read_whole_stream())
    })
}

// fsync related

pub mod fsync {
    use io::{FILERes, FdRes, fd_t};
    use kinds::Copy;
    use libc;
    use ops::Drop;
    use option::{None, Option, Some};
    use os;

    pub enum Level {
        // whatever fsync does on that platform
        FSync,

        // fdatasync on linux, similiar or more on other platforms
        FDataSync,

        // full fsync
        //
        // You must additionally sync the parent directory as well!
        FullFSync,
    }


    // Artifacts that need to fsync on destruction
    pub struct Res<t> {
        arg: Arg<t>,
    }

    #[unsafe_destructor]
    impl<T:Copy> Drop for Res<T> {
        fn finalize(&self) {
            match self.arg.opt_level {
                None => (),
                Some(level) => {
                  // fail hard if not succesful
                  assert!(((self.arg.fsync_fn)(self.arg.val, level)
                    != -1));
                }
            }
        }
    }

    pub fn Res<t: Copy>(arg: Arg<t>) -> Res<t>{
        Res {
            arg: arg
        }
    }

    pub struct Arg<t> {
        val: t,
        opt_level: Option<Level>,
        fsync_fn: @fn(f: t, Level) -> int,
    }

    // fsync file after executing blk
    // FIXME (#2004) find better way to create resources within lifetime of
    // outer res
    pub fn FILE_res_sync(file: &FILERes, opt_level: Option<Level>,
                         blk: &fn(v: Res<*libc::FILE>)) {
        unsafe {
            blk(Res(Arg {
                val: file.f, opt_level: opt_level,
                fsync_fn: |file, l| {
                    unsafe {
                        os::fsync_fd(libc::fileno(file), l) as int
                    }
                }
            }));
        }
    }

    // fsync fd after executing blk
    pub fn fd_res_sync(fd: &FdRes, opt_level: Option<Level>,
                       blk: &fn(v: Res<fd_t>)) {
        blk(Res(Arg {
            val: fd.fd, opt_level: opt_level,
            fsync_fn: |fd, l| os::fsync_fd(fd, l) as int
        }));
    }

    // Type of objects that may want to fsync
    pub trait FSyncable { fn fsync(&self, l: Level) -> int; }

    // Call o.fsync after executing blk
    pub fn obj_sync(o: @FSyncable, opt_level: Option<Level>,
                    blk: &fn(v: Res<@FSyncable>)) {
        blk(Res(Arg {
            val: o, opt_level: opt_level,
            fsync_fn: |o, l| o.fsync(l)
        }));
    }
}

#[cfg(test)]
mod tests {
    use i32;
    use io::{BytesWriter, SeekCur, SeekEnd, SeekSet};
    use io;
    use path::Path;
    use result;
    use str;
    use u64;
    use vec;

    #[test]
    fn test_simple() {
        let tmpfile = &Path("tmp/lib-io-test-simple.tmp");
        debug!(tmpfile);
        let frood: ~str =
            ~"A hoopy frood who really knows where his towel is.";
        debug!(copy frood);
        {
            let out: @io::Writer =
                result::get(
                    &io::file_writer(tmpfile, ~[io::Create, io::Truncate]));
            out.write_str(frood);
        }
        let inp: @io::Reader = result::get(&io::file_reader(tmpfile));
        let frood2: ~str = inp.read_c_str();
        debug!(copy frood2);
        assert!(frood == frood2);
    }

    #[test]
    fn test_readchars_empty() {
        do io::with_str_reader(~"") |inp| {
            let res : ~[char] = inp.read_chars(128);
            assert!((vec::len(res) == 0));
        }
    }

    #[test]
    fn test_read_line_utf8() {
        do io::with_str_reader(~"生锈的汤匙切肉汤hello生锈的汤匙切肉汤") |inp| {
            let line = inp.read_line();
            assert!(line == ~"生锈的汤匙切肉汤hello生锈的汤匙切肉汤");
        }
    }

    #[test]
    fn test_read_lines() {
        do io::with_str_reader(~"a\nb\nc\n") |inp| {
            assert!(inp.read_lines() == ~[~"a", ~"b", ~"c"]);
        }

        do io::with_str_reader(~"a\nb\nc") |inp| {
            assert!(inp.read_lines() == ~[~"a", ~"b", ~"c"]);
        }

        do io::with_str_reader(~"") |inp| {
            assert!(inp.read_lines().is_empty());
        }
    }

    #[test]
    fn test_readchars_wide() {
        let wide_test = ~"生锈的汤匙切肉汤hello生锈的汤匙切肉汤";
        let ivals : ~[int] = ~[
            29983, 38152, 30340, 27748,
            21273, 20999, 32905, 27748,
            104, 101, 108, 108, 111,
            29983, 38152, 30340, 27748,
            21273, 20999, 32905, 27748];
        fn check_read_ln(len : uint, s: &str, ivals: &[int]) {
            do io::with_str_reader(s) |inp| {
                let res : ~[char] = inp.read_chars(len);
                if (len <= vec::len(ivals)) {
                    assert!((vec::len(res) == len));
                }
                assert!(vec::slice(ivals, 0u, vec::len(res)) ==
                             vec::map(res, |x| *x as int));
            }
        }
        let mut i = 0;
        while i < 8 {
            check_read_ln(i, wide_test, ivals);
            i += 1;
        }
        // check a long read for good measure
        check_read_ln(128, wide_test, ivals);
    }

    #[test]
    fn test_readchar() {
        do io::with_str_reader(~"生") |inp| {
            let res : char = inp.read_char();
            assert!((res as int == 29983));
        }
    }

    #[test]
    fn test_readchar_empty() {
        do io::with_str_reader(~"") |inp| {
            let res : char = inp.read_char();
            assert!((res as int == -1));
        }
    }

    #[test]
    fn file_reader_not_exist() {
        match io::file_reader(&Path("not a file")) {
          result::Err(copy e) => {
            assert!(e == ~"error opening not a file");
          }
          result::Ok(_) => fail!()
        }
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(windows))]
    fn test_read_buffer_too_small() {
        let path = &Path("tmp/lib-io-test-read-buffer-too-small.tmp");
        // ensure the file exists
        io::file_writer(path, [io::Create]).get();

        let file = io::file_reader(path).get();
        let mut buf = vec::from_elem(5, 0);
        file.read(buf, 6); // this should fail because buf is too small
    }

    #[test]
    fn test_read_buffer_big_enough() {
        let path = &Path("tmp/lib-io-test-read-buffer-big-enough.tmp");
        // ensure the file exists
        io::file_writer(path, [io::Create]).get();

        let file = io::file_reader(path).get();
        let mut buf = vec::from_elem(5, 0);
        file.read(buf, 4); // this should succeed because buf is big enough
    }

    #[test]
    fn test_write_empty() {
        let file = io::file_writer(&Path("tmp/lib-io-test-write-empty.tmp"),
                                   [io::Create]).get();
        file.write([]);
    }

    #[test]
    fn file_writer_bad_name() {
        match io::file_writer(&Path("?/?"), ~[]) {
          result::Err(copy e) => {
            assert!(str::starts_with(e, "error opening"));
          }
          result::Ok(_) => fail!()
        }
    }

    #[test]
    fn buffered_file_writer_bad_name() {
        match io::buffered_file_writer(&Path("?/?")) {
          result::Err(copy e) => {
            assert!(str::starts_with(e, "error opening"));
          }
          result::Ok(_) => fail!()
        }
    }

    #[test]
    fn bytes_buffer_overwrite() {
        let wr = BytesWriter();
        wr.write(~[0u8, 1u8, 2u8, 3u8]);
        assert!(wr.bytes == ~[0u8, 1u8, 2u8, 3u8]);
        wr.seek(-2, SeekCur);
        wr.write(~[4u8, 5u8, 6u8, 7u8]);
        assert!(wr.bytes == ~[0u8, 1u8, 4u8, 5u8, 6u8, 7u8]);
        wr.seek(-2, SeekEnd);
        wr.write(~[8u8]);
        wr.seek(1, SeekSet);
        wr.write(~[9u8]);
        assert!(wr.bytes == ~[0u8, 9u8, 4u8, 5u8, 8u8, 7u8]);
    }

    #[test]
    fn test_read_write_le() {
        let path = Path("tmp/lib-io-test-read-write-le.tmp");
        let uints = [0, 1, 2, 42, 10_123, 100_123_456, u64::max_value];

        // write the ints to the file
        {
            let file = io::file_writer(&path, [io::Create]).get();
            for uints.each |i| {
                file.write_le_u64(*i);
            }
        }

        // then read them back and check that they are the same
        {
            let file = io::file_reader(&path).get();
            for uints.each |i| {
                assert!(file.read_le_u64() == *i);
            }
        }
    }

    #[test]
    fn test_read_write_be() {
        let path = Path("tmp/lib-io-test-read-write-be.tmp");
        let uints = [0, 1, 2, 42, 10_123, 100_123_456, u64::max_value];

        // write the ints to the file
        {
            let file = io::file_writer(&path, [io::Create]).get();
            for uints.each |i| {
                file.write_be_u64(*i);
            }
        }

        // then read them back and check that they are the same
        {
            let file = io::file_reader(&path).get();
            for uints.each |i| {
                assert!(file.read_be_u64() == *i);
            }
        }
    }

    #[test]
    fn test_read_be_int_n() {
        let path = Path("tmp/lib-io-test-read-be-int-n.tmp");
        let ints = [i32::min_value, -123456, -42, -5, 0, 1, i32::max_value];

        // write the ints to the file
        {
            let file = io::file_writer(&path, [io::Create]).get();
            for ints.each |i| {
                file.write_be_i32(*i);
            }
        }

        // then read them back and check that they are the same
        {
            let file = io::file_reader(&path).get();
            for ints.each |i| {
                // this tests that the sign extension is working
                // (comparing the values as i32 would not test this)
                assert!(file.read_be_int_n(4) == *i as i64);
            }
        }
    }

    #[test]
    fn test_read_f32() {
        let path = Path("tmp/lib-io-test-read-f32.tmp");
        //big-endian floating-point 8.1250
        let buf = ~[0x41, 0x02, 0x00, 0x00];

        {
            let file = io::file_writer(&path, [io::Create]).get();
            file.write(buf);
        }

        {
            let file = io::file_reader(&path).get();
            let f = file.read_be_f32();
            assert!(f == 8.1250);
        }
    }

#[test]
    fn test_read_write_f32() {
        let path = Path("tmp/lib-io-test-read-write-f32.tmp");
        let f:f32 = 8.1250;

        {
            let file = io::file_writer(&path, [io::Create]).get();
            file.write_be_f32(f);
            file.write_le_f32(f);
        }

        {
            let file = io::file_reader(&path).get();
            assert!(file.read_be_f32() == 8.1250);
            assert!(file.read_le_f32() == 8.1250);
        }
    }
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
