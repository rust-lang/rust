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

The `io` module contains basic input and output routines.

A quick summary:

## `Reader` and `Writer` traits

These traits define the minimal set of methods that anything that can do
input and output should implement.

## `ReaderUtil` and `WriterUtil` traits

Richer methods that allow you to do more. `Reader` only lets you read a certain
number of bytes into a buffer, while `ReaderUtil` allows you to read a whole
line, for example.

Generally, these richer methods are probably the ones you want to actually
use in day-to-day Rust.

Furthermore, because there is an implementation of `ReaderUtil` for
`<T: Reader>`, when your input or output code implements `Reader`, you get
all of these methods for free.

## `print` and `println`

These very useful functions are defined here. You generally don't need to
import them, though, as the prelude already does.

## `stdin`, `stdout`, and `stderr`

These functions return references to the classic three file descriptors. They
implement `Reader` and `Writer`, where appropriate.

*/

#[allow(missing_doc)];

use cast;
use cast::transmute;
use clone::Clone;
use c_str::ToCStr;
use container::Container;
use int;
use iter::Iterator;
use libc::consts::os::posix88::*;
use libc::{c_int, c_void, size_t};
use libc;
use num;
use ops::Drop;
use option::{Some, None};
use os;
use path::Path;
use ptr;
use result::{Result, Ok, Err};
use str::{StrSlice, OwnedStr};
use str;
use to_str::ToStr;
use uint;
use vec::{MutableVector, ImmutableVector, OwnedVector, OwnedCopyableVector, CopyableVector};
use vec;

#[allow(non_camel_case_types)] // not sure what to do about this
pub type fd_t = c_int;

pub mod rustrt {
    use libc;

    #[abi = "cdecl"]
    #[link_name = "rustrt"]
    extern {
        pub fn rust_get_stdin() -> *libc::FILE;
        pub fn rust_get_stdout() -> *libc::FILE;
        pub fn rust_get_stderr() -> *libc::FILE;
    }
}

// Reading

// FIXME (#2004): This is all buffered. We might need an unbuffered variant
// as well
/**
* The SeekStyle enum describes the relationship between the position
* we'd like to seek to from our current position. It's used as an argument
* to the `seek` method defined on the `Reader` trait.
*
* There are three seek styles:
*
* 1. `SeekSet` means that the new position should become our position.
* 2. `SeekCur` means that we should seek from the current position.
* 3. `SeekEnd` means that we should seek from the end.
*
* # Examples
*
* None right now.
*/
pub enum SeekStyle { SeekSet, SeekEnd, SeekCur, }


/**
* The core Reader trait. All readers must implement this trait.
*
* # Examples
*
* None right now.
*/
pub trait Reader {
    // FIXME (#2004): Seekable really should be orthogonal.

    // FIXME (#2982): This should probably return an error.
    /**
    * Reads bytes and puts them into `bytes`, advancing the cursor. Returns the
    * number of bytes read.
    *
    * The number of bytes to be read is `len` or the end of the file,
    * whichever comes first.
    *
    * The buffer must be at least `len` bytes long.
    *
    * `read` is conceptually similar to C's `fread` function.
    *
    * # Examples
    *
    * None right now.
    */
    fn read(&self, bytes: &mut [u8], len: uint) -> uint;

    /**
    * Reads a single byte, advancing the cursor.
    *
    * In the case of an EOF or an error, returns a negative value.
    *
    * `read_byte` is conceptually similar to C's `getc` function.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_byte(&self) -> int;

    /**
    * Returns a boolean value: are we currently at EOF?
    *
    * Note that stream position may be already at the end-of-file point,
    * but `eof` returns false until an attempt to read at that position.
    *
    * `eof` is conceptually similar to C's `feof` function.
    *
    * # Examples
    *
    * None right now.
    */
    fn eof(&self) -> bool;

    /**
    * Seek to a given `position` in the stream.
    *
    * Takes an optional SeekStyle, which affects how we seek from the
    * position. See `SeekStyle` docs for more details.
    *
    * `seek` is conceptually similar to C's `fseek` function.
    *
    * # Examples
    *
    * None right now.
    */
    fn seek(&self, position: int, style: SeekStyle);

    /**
    * Returns the current position within the stream.
    *
    * `tell` is conceptually similar to C's `ftell` function.
    *
    * # Examples
    *
    * None right now.
    */
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

/**
* The `ReaderUtil` trait is a home for many of the utility functions
* a particular Reader should implement.
*
* The default `Reader` trait is focused entirely on bytes. `ReaderUtil` is based
* on higher-level concepts like 'chars' and 'lines.'
*
* # Examples:
*
* None right now.
*/
pub trait ReaderUtil {

    /**
    * Reads `len` number of bytes, and gives you a new vector back.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_bytes(&self, len: uint) -> ~[u8];

    /**
    * Reads up until a specific byte is seen or EOF.
    *
    * The `include` parameter specifies if the character should be included
    * in the returned string.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_until(&self, c: u8, include: bool) -> ~str;

    /**
    * Reads up until the first '\n' or EOF.
    *
    * The '\n' is not included in the result.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_line(&self) -> ~str;

    /**
    * Reads `n` chars.
    *
    * Assumes that those chars are UTF-8 encoded.
    *
    * The '\n' is not included in the result.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_chars(&self, n: uint) -> ~[char];

    /**
    * Reads a single UTF-8 encoded char.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_char(&self) -> char;

    /**
    * Reads up until the first null byte or EOF.
    *
    * The null byte is not returned.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_c_str(&self) -> ~str;

    /**
    * Reads all remaining data in the stream.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_whole_stream(&self) -> ~[u8];

    /**
    * Iterate over every byte until EOF or the iterator breaks.
    *
    * # Examples
    *
    * None right now.
    */
    fn each_byte(&self, it: &fn(int) -> bool) -> bool;

    /**
    * Iterate over every char until EOF or the iterator breaks.
    *
    * # Examples
    *
    * None right now.
    */
    fn each_char(&self, it: &fn(char) -> bool) -> bool;

    /**
    * Iterate over every line until EOF or the iterator breaks.
    *
    * # Examples
    *
    * None right now.
    */
    fn each_line(&self, it: &fn(&str) -> bool) -> bool;

    /**
    * Reads all of the lines in the stream.
    *
    * Returns a vector of those lines.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_lines(&self) -> ~[~str];

    /**
    * Reads `n` little-endian unsigned integer bytes.
    *
    * `n` must be between 1 and 8, inclusive.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_le_uint_n(&self, nbytes: uint) -> u64;

    /**
    * Reads `n` little-endian signed integer bytes.
    *
    * `n` must be between 1 and 8, inclusive.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_le_int_n(&self, nbytes: uint) -> i64;

    /**
    * Reads `n` big-endian unsigned integer bytes.
    *
    * `n` must be between 1 and 8, inclusive.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_be_uint_n(&self, nbytes: uint) -> u64;

    /**
    * Reads `n` big-endian signed integer bytes.
    *
    * `n` must be between 1 and 8, inclusive.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_be_int_n(&self, nbytes: uint) -> i64;

    /**
    * Reads a little-endian unsigned integer.
    *
    * The number of bytes returned is system-dependant.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_le_uint(&self) -> uint;

    /**
    * Reads a little-endian integer.
    *
    * The number of bytes returned is system-dependant.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_le_int(&self) -> int;

    /**
    * Reads a big-endian unsigned integer.
    *
    * The number of bytes returned is system-dependant.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_be_uint(&self) -> uint;

    /**
    * Reads a big-endian integer.
    *
    * The number of bytes returned is system-dependant.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_be_int(&self) -> int;

    /**
    * Reads a big-endian `u64`.
    *
    * `u64`s are 8 bytes long.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_be_u64(&self) -> u64;

    /**
    * Reads a big-endian `u32`.
    *
    * `u32`s are 4 bytes long.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_be_u32(&self) -> u32;

    /**
    * Reads a big-endian `u16`.
    *
    * `u16`s are 2 bytes long.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_be_u16(&self) -> u16;

    /**
    * Reads a big-endian `i64`.
    *
    * `i64`s are 8 bytes long.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_be_i64(&self) -> i64;

    /**
    * Reads a big-endian `i32`.
    *
    * `i32`s are 4 bytes long.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_be_i32(&self) -> i32;

    /**
    * Reads a big-endian `i16`.
    *
    * `i16`s are 2 bytes long.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_be_i16(&self) -> i16;

    /**
    * Reads a big-endian `f64`.
    *
    * `f64`s are 8 byte, IEEE754 double-precision floating point numbers.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_be_f64(&self) -> f64;

    /**
    * Reads a big-endian `f32`.
    *
    * `f32`s are 4 byte, IEEE754 single-precision floating point numbers.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_be_f32(&self) -> f32;

    /**
    * Reads a little-endian `u64`.
    *
    * `u64`s are 8 bytes long.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_le_u64(&self) -> u64;

    /**
    * Reads a little-endian `u32`.
    *
    * `u32`s are 4 bytes long.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_le_u32(&self) -> u32;

    /**
    * Reads a little-endian `u16`.
    *
    * `u16`s are 2 bytes long.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_le_u16(&self) -> u16;

    /**
    * Reads a little-endian `i64`.
    *
    * `i64`s are 8 bytes long.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_le_i64(&self) -> i64;

    /**
    * Reads a little-endian `i32`.
    *
    * `i32`s are 4 bytes long.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_le_i32(&self) -> i32;

    /**
    * Reads a little-endian `i16`.
    *
    * `i16`s are 2 bytes long.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_le_i16(&self) -> i16;

    /**
    * Reads a little-endian `f64`.
    *
    * `f64`s are 8 byte, IEEE754 double-precision floating point numbers.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_le_f64(&self) -> f64;

    /**
    * Reads a little-endian `f32`.
    *
    * `f32`s are 4 byte, IEEE754 single-precision floating point numbers.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_le_f32(&self) -> f32;

    /**
    * Read a u8.
    *
    * `u8`s are 1 byte.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_u8(&self) -> u8;

    /**
    * Read an i8.
    *
    * `i8`s are 1 byte.
    *
    * # Examples
    *
    * None right now.
    */
    fn read_i8(&self) -> i8;
}

impl<T:Reader> ReaderUtil for T {

    fn read_bytes(&self, len: uint) -> ~[u8] {
        let mut bytes = vec::with_capacity(len);
        unsafe { vec::raw::set_len(&mut bytes, len); }

        let count = self.read(bytes, len);

        unsafe { vec::raw::set_len(&mut bytes, count); }
        bytes
    }

    fn read_until(&self, c: u8, include: bool) -> ~str {
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
        str::from_utf8(bytes)
    }

    fn read_line(&self) -> ~str {
        self.read_until('\n' as u8, false)
    }

    fn read_chars(&self, n: uint) -> ~[char] {
        // returns the (consumed offset, n_req), appends characters to &chars
        fn chars_from_utf8<T:Reader>(bytes: &~[u8], chars: &mut ~[char])
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
                    unsafe {
                        chars.push(transmute(b0 as u32));
                    }
                    continue;
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
                    assert_eq!(next & 192, 128);
                    val <<= 6;
                    val += (next & 63) as uint;
                }
                // See str::StrSlice::char_at
                val += ((b0 << ((w + 1) as u8)) as uint)
                    << (w - 1) * 6 - w - 1u;
                unsafe {
                    chars.push(transmute(val as u32));
                }
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
            let (offset, nbreq) = chars_from_utf8::<T>(&bytes, &mut chars);
            let ncreq = n - chars.len();
            // again we either know we need a certain number of bytes
            // to complete a character, or we make sure we don't
            // over-read by reading 1-byte per char needed
            nbread = if ncreq > nbreq { ncreq } else { nbreq };
            if nbread > 0 {
                bytes = bytes.slice(offset, bytes.len()).to_owned();
            }
        }
        chars
    }

    fn read_char(&self) -> char {
        let c = self.read_chars(1);
        if c.len() == 0 {
            return unsafe { transmute(-1u32) }; // FIXME: #8971: unsound
        }
        assert_eq!(c.len(), 1);
        return c[0];
    }

    fn read_c_str(&self) -> ~str {
        self.read_until(0u8, false)
    }

    fn read_whole_stream(&self) -> ~[u8] {
        let mut bytes: ~[u8] = ~[];
        while !self.eof() { bytes.push_all(self.read_bytes(2048u)); }
        bytes
    }

    fn each_byte(&self, it: &fn(int) -> bool) -> bool {
        loop {
            match self.read_byte() {
                -1 => break,
                ch => if !it(ch) { return false; }
            }
        }
        return true;
    }

    fn each_char(&self, it: &fn(char) -> bool) -> bool {
        // FIXME: #8971: unsound
        let eof: char = unsafe { transmute(-1u32) };
        loop {
            match self.read_char() {
                c if c == eof => break,
                ch => if !it(ch) { return false; }
            }
        }
        return true;
    }

    fn each_line(&self, it: &fn(s: &str) -> bool) -> bool {
        while !self.eof() {
            // include the \n, so that we can distinguish an entirely empty
            // line read after "...\n", and the trailing empty line in
            // "...\n\n".
            let mut line = self.read_until('\n' as u8, true);

            // blank line at the end of the reader is ignored
            if self.eof() && line.is_empty() { break; }

            // trim the \n, so that each_line is consistent with read_line
            let n = line.len();
            if line[n-1] == '\n' as u8 {
                unsafe { str::raw::set_len(&mut line, n-1); }
            }

            if !it(line) { return false; }
        }
        return true;
    }

    fn read_lines(&self) -> ~[~str] {
        do vec::build(None) |push| {
            do self.each_line |line| {
                push(line.to_owned());
                true
            };
        }
    }

    // FIXME int reading methods need to deal with eof - issue #2004

    fn read_le_uint_n(&self, nbytes: uint) -> u64 {
        assert!(nbytes > 0 && nbytes <= 8);

        let mut val = 0u64;
        let mut pos = 0;
        let mut i = nbytes;
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

        let mut val = 0u64;
        let mut i = nbytes;
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
        #[fixed_stack_segment]; #[inline(never)];

        unsafe {
            do bytes.as_mut_buf |buf_p, buf_len| {
                assert!(buf_len >= len);

                let count = libc::fread(buf_p as *mut c_void, 1u as size_t,
                                        len as size_t, *self) as uint;
                if count < len {
                  match libc::ferror(*self) {
                    0 => (),
                    _ => {
                      error2!("error reading buffer: {}", os::last_os_error());
                      fail2!();
                    }
                  }
                }

                count
            }
        }
    }
    fn read_byte(&self) -> int {
        #[fixed_stack_segment]; #[inline(never)];

        unsafe {
            libc::fgetc(*self) as int
        }
    }
    fn eof(&self) -> bool {
        #[fixed_stack_segment]; #[inline(never)];

        unsafe {
            return libc::feof(*self) != 0 as c_int;
        }
    }
    fn seek(&self, offset: int, whence: SeekStyle) {
        #[fixed_stack_segment]; #[inline(never)];

        unsafe {
            assert!(libc::fseek(*self,
                                     offset as libc::c_long,
                                     convert_whence(whence)) == 0 as c_int);
        }
    }
    fn tell(&self) -> uint {
        #[fixed_stack_segment]; #[inline(never)];

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

impl FILERes {
    pub fn new(f: *libc::FILE) -> FILERes {
        FILERes { f: f }
    }
}

impl Drop for FILERes {
    fn drop(&mut self) {
        #[fixed_stack_segment]; #[inline(never)];

        unsafe {
            libc::fclose(self.f);
        }
    }
}

pub fn FILE_reader(f: *libc::FILE, cleanup: bool) -> @Reader {
    if cleanup {
        @Wrapper { base: f, cleanup: FILERes::new(f) } as @Reader
    } else {
        @f as @Reader
    }
}

// FIXME (#2004): this should either be an trait-less impl, a set of
// top-level functions that take a reader, or a set of default methods on
// reader (which can then be called reader)

/**
* Gives a `Reader` that allows you to read values from standard input.
*
* # Example
*
* ```rust
* let stdin = std::io::stdin();
* let line = stdin.read_line();
* std::io::print(line);
* ```
*/
pub fn stdin() -> @Reader {
    #[fixed_stack_segment]; #[inline(never)];

    unsafe {
        @rustrt::rust_get_stdin() as @Reader
    }
}

pub fn file_reader(path: &Path) -> Result<@Reader, ~str> {
    #[fixed_stack_segment]; #[inline(never)];

    let f = do path.with_c_str |pathbuf| {
        do "rb".with_c_str |modebuf| {
            unsafe { libc::fopen(pathbuf, modebuf as *libc::c_char) }
        }
    };

    if f as uint == 0u {
        Err(~"error opening " + path.to_str())
    } else {
        Ok(FILE_reader(f, true))
    }
}


// Byte readers
pub struct BytesReader {
    // FIXME(#5723) see other FIXME below
    // FIXME(#7268) this should also be parameterized over <'self>
    bytes: &'static [u8],
    pos: @mut uint
}

impl Reader for BytesReader {
    fn read(&self, bytes: &mut [u8], len: uint) -> uint {
        let count = num::min(len, self.bytes.len() - *self.pos);

        let view = self.bytes.slice(*self.pos, self.bytes.len());
        vec::bytes::copy_memory(bytes, view, count);

        *self.pos += count;

        count
    }

    fn read_byte(&self) -> int {
        if *self.pos == self.bytes.len() {
            return -1;
        }

        let b = self.bytes[*self.pos];
        *self.pos += 1u;
        b as int
    }

    fn eof(&self) -> bool {
        *self.pos == self.bytes.len()
    }

    fn seek(&self, offset: int, whence: SeekStyle) {
        let pos = *self.pos;
        *self.pos = seek_in_buf(offset, pos, self.bytes.len(), whence);
    }

    fn tell(&self) -> uint {
        *self.pos
    }
}

pub fn with_bytes_reader<T>(bytes: &[u8], f: &fn(@Reader) -> T) -> T {
    // XXX XXX XXX this is glaringly unsound
    // FIXME(#5723) Use a &Reader for the callback's argument. Should be:
    // fn with_bytes_reader<'r, T>(bytes: &'r [u8], f: &fn(&'r Reader) -> T) -> T
    let bytes: &'static [u8] = unsafe { cast::transmute(bytes) };
    f(@BytesReader {
        bytes: bytes,
        pos: @mut 0
    } as @Reader)
}

pub fn with_str_reader<T>(s: &str, f: &fn(@Reader) -> T) -> T {
    // FIXME(#5723): As above.
    with_bytes_reader(s.as_bytes(), f)
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
    fn write(&self, v: &[u8]);

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
    fn write(&self, v: &[u8]) { self.write(v) }
    fn seek(&self, a: int, b: SeekStyle) { self.seek(a, b) }
    fn tell(&self) -> uint { self.tell() }
    fn flush(&self) -> int { self.flush() }
    fn get_type(&self) -> WriterType { self.get_type() }
}

impl<W:Writer,C> Writer for Wrapper<W, C> {
    fn write(&self, bs: &[u8]) { self.base.write(bs); }
    fn seek(&self, off: int, style: SeekStyle) { self.base.seek(off, style); }
    fn tell(&self) -> uint { self.base.tell() }
    fn flush(&self) -> int { self.base.flush() }
    fn get_type(&self) -> WriterType { File }
}

impl Writer for *libc::FILE {
    fn write(&self, v: &[u8]) {
        #[fixed_stack_segment]; #[inline(never)];

        unsafe {
            do v.as_imm_buf |vbuf, len| {
                let nout = libc::fwrite(vbuf as *c_void,
                                        1,
                                        len as size_t,
                                        *self);
                if nout != len as size_t {
                    error2!("error writing buffer: {}", os::last_os_error());
                    fail2!();
                }
            }
        }
    }
    fn seek(&self, offset: int, whence: SeekStyle) {
        #[fixed_stack_segment]; #[inline(never)];

        unsafe {
            assert!(libc::fseek(*self,
                                     offset as libc::c_long,
                                     convert_whence(whence)) == 0 as c_int);
        }
    }
    fn tell(&self) -> uint {
        #[fixed_stack_segment]; #[inline(never)];

        unsafe {
            libc::ftell(*self) as uint
        }
    }
    fn flush(&self) -> int {
        #[fixed_stack_segment]; #[inline(never)];

        unsafe {
            libc::fflush(*self) as int
        }
    }
    fn get_type(&self) -> WriterType {
        #[fixed_stack_segment]; #[inline(never)];

        unsafe {
            let fd = libc::fileno(*self);
            if libc::isatty(fd) == 0 { File   }
            else                     { Screen }
        }
    }
}

impl Writer for fd_t {
    fn write(&self, v: &[u8]) {
        #[fixed_stack_segment]; #[inline(never)];

        #[cfg(windows)]
        type IoSize = libc::c_uint;
        #[cfg(windows)]
        type IoRet = c_int;

        #[cfg(unix)]
        type IoSize = size_t;
        #[cfg(unix)]
        type IoRet = libc::ssize_t;

        unsafe {
            let mut count = 0u;
            do v.as_imm_buf |vbuf, len| {
                while count < len {
                    let vb = ptr::offset(vbuf, count as int) as *c_void;
                    let nout = libc::write(*self, vb, len as IoSize);
                    if nout < 0 as IoRet {
                        error2!("error writing buffer: {}", os::last_os_error());
                        fail2!();
                    }
                    count += nout as uint;
                }
            }
        }
    }
    fn seek(&self, _offset: int, _whence: SeekStyle) {
        error2!("need 64-bit foreign calls for seek, sorry");
        fail2!();
    }
    fn tell(&self) -> uint {
        error2!("need 64-bit foreign calls for tell, sorry");
        fail2!();
    }
    fn flush(&self) -> int { 0 }
    fn get_type(&self) -> WriterType {
        #[fixed_stack_segment]; #[inline(never)];

        unsafe {
            if libc::isatty(*self) == 0 { File } else { Screen }
        }
    }
}

pub struct FdRes {
    fd: fd_t,
}

impl FdRes {
    pub fn new(fd: fd_t) -> FdRes {
        FdRes { fd: fd }
    }
}

impl Drop for FdRes {
    fn drop(&mut self) {
        #[fixed_stack_segment]; #[inline(never)];

        unsafe {
            libc::close(self.fd);
        }
    }
}

pub fn fd_writer(fd: fd_t, cleanup: bool) -> @Writer {
    if cleanup {
        @Wrapper { base: fd, cleanup: FdRes::new(fd) } as @Writer
    } else {
        @fd as @Writer
    }
}


pub fn mk_file_writer(path: &Path, flags: &[FileFlag])
                   -> Result<@Writer, ~str> {
    #[fixed_stack_segment]; #[inline(never)];

    #[cfg(windows)]
    fn wb() -> c_int {
      (O_WRONLY | libc::consts::os::extra::O_BINARY) as c_int
    }

    #[cfg(unix)]
    fn wb() -> c_int { O_WRONLY as c_int }

    let mut fflags: c_int = wb();
    for f in flags.iter() {
        match *f {
          Append => fflags |= O_APPEND as c_int,
          Create => fflags |= O_CREAT as c_int,
          Truncate => fflags |= O_TRUNC as c_int,
          NoFlag => ()
        }
    }
    let fd = unsafe {
        do path.with_c_str |pathbuf| {
            libc::open(pathbuf, fflags, (S_IRUSR | S_IWUSR) as c_int)
        }
    };
    if fd < (0 as c_int) {
        Err(format!("error opening {}: {}", path.to_str(), os::last_os_error()))
    } else {
        Ok(fd_writer(fd, true))
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

        let mut bytes: ~[u8] = ~[];
        let mut i = size;
        let mut n = n;
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

pub fn u64_from_be_bytes(data: &[u8],
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

    /// Write a little-endian IEEE754 single-precision floating-point
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
    fn write_str(&self, s: &str) { self.write(s.as_bytes()) }
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

pub fn file_writer(path: &Path, flags: &[FileFlag]) -> Result<@Writer, ~str> {
    mk_file_writer(path, flags).and_then(|w| Ok(w))
}

// FIXME (#2004) it would be great if this could be a const
// FIXME (#2004) why are these different from the way stdin() is
// implemented?


/**
* Gives a `Writer` which allows you to write to the standard output.
*
* # Example
*
* ```rust
* let stdout = std::io::stdout();
* stdout.write_str("hello\n");
* ```
*/
pub fn stdout() -> @Writer { fd_writer(libc::STDOUT_FILENO as c_int, false) }

/**
* Gives a `Writer` which allows you to write to standard error.
*
* # Example
*
* ```rust
* let stderr = std::io::stderr();
* stderr.write_str("hello\n");
* ```
*/
pub fn stderr() -> @Writer { fd_writer(libc::STDERR_FILENO as c_int, false) }

/**
* Prints a string to standard output.
*
* This string will not have an implicit newline at the end. If you want
* an implicit newline, please see `println`.
*
* # Example
*
* ```rust
* // print is imported into the prelude, and so is always available.
* print("hello");
* ```
*/
pub fn print(s: &str) {
    stdout().write_str(s);
}

/**
* Prints a string to standard output, followed by a newline.
*
* If you do not want an implicit newline, please see `print`.
*
* # Example
*
* ```rust
* // println is imported into the prelude, and so is always available.
* println("hello");
* ```
*/
pub fn println(s: &str) {
    stdout().write_line(s);
}

pub struct BytesWriter {
    bytes: @mut ~[u8],
    pos: @mut uint,
}

impl BytesWriter {
    pub fn new() -> BytesWriter {
        BytesWriter {
            bytes: @mut ~[],
            pos: @mut 0
        }
    }
}

impl Writer for BytesWriter {
    fn write(&self, v: &[u8]) {
        let v_len = v.len();

        let bytes = &mut *self.bytes;
        let count = num::max(bytes.len(), *self.pos + v_len);
        bytes.reserve(count);

        unsafe {
            vec::raw::set_len(bytes, count);

            let view = bytes.mut_slice(*self.pos, count);
            vec::bytes::copy_memory(view, v, v_len);
        }

        *self.pos += v_len;
    }

    fn seek(&self, offset: int, whence: SeekStyle) {
        let pos = *self.pos;
        let len = self.bytes.len();
        *self.pos = seek_in_buf(offset, pos, len, whence);
    }

    fn tell(&self) -> uint {
        *self.pos
    }

    fn flush(&self) -> int {
        0
    }

    fn get_type(&self) -> WriterType {
        File
    }
}

pub fn with_bytes_writer(f: &fn(@Writer)) -> ~[u8] {
    let wr = @BytesWriter::new();
    f(wr as @Writer);
    let @BytesWriter { bytes, _ } = wr;
    (*bytes).clone()
}

pub fn with_str_writer(f: &fn(@Writer)) -> ~str {
    str::from_utf8(with_bytes_writer(f))
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

pub fn read_whole_file_str(file: &Path) -> Result<~str, ~str> {
    do read_whole_file(file).and_then |bytes| {
        if str::is_utf8(bytes) {
            Ok(str::from_utf8(bytes))
        } else {
            Err(file.to_str() + " is not UTF-8")
        }
    }
}

// FIXME (#2004): implement this in a low-level way. Going through the
// abstractions is pointless.
pub fn read_whole_file(file: &Path) -> Result<~[u8], ~str> {
    do file_reader(file).and_then |rdr| {
        Ok(rdr.read_whole_stream())
    }
}

// fsync related

pub mod fsync {
    use io::{FILERes, FdRes, fd_t};
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

    impl <t> Res<t> {
        pub fn new(arg: Arg<t>) -> Res<t> {
            Res { arg: arg }
        }
    }

    #[unsafe_destructor]
    impl<T> Drop for Res<T> {
        fn drop(&mut self) {
            match self.arg.opt_level {
                None => (),
                Some(level) => {
                  // fail hard if not succesful
                  assert!(((self.arg.fsync_fn)(&self.arg.val, level) != -1));
                }
            }
        }
    }

    pub struct Arg<t> {
        val: t,
        opt_level: Option<Level>,
        fsync_fn: extern "Rust" fn(f: &t, Level) -> int,
    }

    // fsync file after executing blk
    // FIXME (#2004) find better way to create resources within lifetime of
    // outer res
    pub fn FILE_res_sync(file: &FILERes,
                         opt_level: Option<Level>,
                         blk: &fn(v: Res<*libc::FILE>)) {
        blk(Res::new(Arg {
            val: file.f,
            opt_level: opt_level,
            fsync_fn: fsync_FILE,
        }));

        fn fileno(stream: *libc::FILE) -> libc::c_int {
            #[fixed_stack_segment]; #[inline(never)];
            unsafe { libc::fileno(stream) }
        }

        fn fsync_FILE(stream: &*libc::FILE, level: Level) -> int {
            fsync_fd(fileno(*stream), level)
        }
    }

    // fsync fd after executing blk
    pub fn fd_res_sync(fd: &FdRes, opt_level: Option<Level>,
                       blk: &fn(v: Res<fd_t>)) {
        blk(Res::new(Arg {
            val: fd.fd,
            opt_level: opt_level,
            fsync_fn: fsync_fd_helper,
        }));
    }

    fn fsync_fd(fd: libc::c_int, level: Level) -> int {
        #[fixed_stack_segment]; #[inline(never)];

        os::fsync_fd(fd, level) as int
    }

    fn fsync_fd_helper(fd_ptr: &libc::c_int, level: Level) -> int {
        fsync_fd(*fd_ptr, level)
    }

    // Type of objects that may want to fsync
    pub trait FSyncable { fn fsync(&self, l: Level) -> int; }

    // Call o.fsync after executing blk
    pub fn obj_sync(o: @FSyncable, opt_level: Option<Level>,
                    blk: &fn(v: Res<@FSyncable>)) {
        blk(Res::new(Arg {
            val: o,
            opt_level: opt_level,
            fsync_fn: obj_fsync_fn,
        }));
    }

    fn obj_fsync_fn(o: &@FSyncable, level: Level) -> int {
        (*o).fsync(level)
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;
    use i32;
    use io::{BytesWriter, SeekCur, SeekEnd, SeekSet};
    use io;
    use path::Path;
    use result::{Ok, Err};
    use u64;
    use vec;
    use cast::transmute;

    #[test]
    fn test_simple() {
        let tmpfile = &Path("tmp/lib-io-test-simple.tmp");
        debug2!("{:?}", tmpfile);
        let frood: ~str =
            ~"A hoopy frood who really knows where his towel is.";
        debug2!("{}", frood.clone());
        {
            let out = io::file_writer(tmpfile, [io::Create, io::Truncate]).unwrap();
            out.write_str(frood);
        }
        let inp = io::file_reader(tmpfile).unwrap();
        let frood2: ~str = inp.read_c_str();
        debug2!("{}", frood2.clone());
        assert_eq!(frood, frood2);
    }

    #[test]
    fn test_each_byte_each_char_file() {
        // Issue #5056 -- shouldn't include trailing EOF.
        let path = Path("tmp/lib-io-test-each-byte-each-char-file.tmp");

        {
            // create empty, enough to reproduce a problem
            io::file_writer(&path, [io::Create]).unwrap();
        }

        {
            let file = io::file_reader(&path).unwrap();
            do file.each_byte() |_| {
                fail2!("must be empty")
            };
        }

        {
            let file = io::file_reader(&path).unwrap();
            do file.each_char() |_| {
                fail2!("must be empty")
            };
        }
    }

    #[test]
    fn test_readchars_empty() {
        do io::with_str_reader("") |inp| {
            let res : ~[char] = inp.read_chars(128);
            assert_eq!(res.len(), 0);
        }
    }

    #[test]
    fn test_read_line_utf8() {
        do io::with_str_reader("生锈的汤匙切肉汤hello生锈的汤匙切肉汤") |inp| {
            let line = inp.read_line();
            assert_eq!(line, ~"生锈的汤匙切肉汤hello生锈的汤匙切肉汤");
        }
    }

    #[test]
    fn test_read_lines() {
        do io::with_str_reader("a\nb\nc\n") |inp| {
            assert_eq!(inp.read_lines(), ~[~"a", ~"b", ~"c"]);
        }

        do io::with_str_reader("a\nb\nc") |inp| {
            assert_eq!(inp.read_lines(), ~[~"a", ~"b", ~"c"]);
        }

        do io::with_str_reader("") |inp| {
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
                if len <= ivals.len() {
                    assert_eq!(res.len(), len);
                }
                for (iv, c) in ivals.iter().zip(res.iter()) {
                    assert!(*iv == *c as int)
                }
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
        do io::with_str_reader("生") |inp| {
            let res = inp.read_char();
            assert_eq!(res as int, 29983);
        }
    }

    #[test]
    fn test_readchar_empty() {
        do io::with_str_reader("") |inp| {
            let res = inp.read_char();
            assert_eq!(res, unsafe { transmute(-1u32) }); // FIXME: #8971: unsound
        }
    }

    #[test]
    fn file_reader_not_exist() {
        match io::file_reader(&Path("not a file")) {
          Err(e) => {
            assert_eq!(e, ~"error opening not a file");
          }
          Ok(_) => fail2!()
        }
    }

    #[test]
    #[should_fail]
    fn test_read_buffer_too_small() {
        let path = &Path("tmp/lib-io-test-read-buffer-too-small.tmp");
        // ensure the file exists
        io::file_writer(path, [io::Create]).unwrap();

        let file = io::file_reader(path).unwrap();
        let mut buf = vec::from_elem(5, 0u8);
        file.read(buf, 6); // this should fail because buf is too small
    }

    #[test]
    fn test_read_buffer_big_enough() {
        let path = &Path("tmp/lib-io-test-read-buffer-big-enough.tmp");
        // ensure the file exists
        io::file_writer(path, [io::Create]).unwrap();

        let file = io::file_reader(path).unwrap();
        let mut buf = vec::from_elem(5, 0u8);
        file.read(buf, 4); // this should succeed because buf is big enough
    }

    #[test]
    fn test_write_empty() {
        let file = io::file_writer(&Path("tmp/lib-io-test-write-empty.tmp"),
                                   [io::Create]).unwrap();
        file.write([]);
    }

    #[test]
    fn file_writer_bad_name() {
        match io::file_writer(&Path("?/?"), []) {
          Err(e) => {
            assert!(e.starts_with("error opening"));
          }
          Ok(_) => fail2!()
        }
    }

    #[test]
    fn bytes_buffer_overwrite() {
        let wr = BytesWriter::new();
        wr.write([0u8, 1u8, 2u8, 3u8]);
        assert!(*wr.bytes == ~[0u8, 1u8, 2u8, 3u8]);
        wr.seek(-2, SeekCur);
        wr.write([4u8, 5u8, 6u8, 7u8]);
        assert!(*wr.bytes == ~[0u8, 1u8, 4u8, 5u8, 6u8, 7u8]);
        wr.seek(-2, SeekEnd);
        wr.write([8u8]);
        wr.seek(1, SeekSet);
        wr.write([9u8]);
        assert!(*wr.bytes == ~[0u8, 9u8, 4u8, 5u8, 8u8, 7u8]);
    }

    #[test]
    fn test_read_write_le() {
        let path = Path("tmp/lib-io-test-read-write-le.tmp");
        let uints = [0, 1, 2, 42, 10_123, 100_123_456, u64::max_value];

        // write the ints to the file
        {
            let file = io::file_writer(&path, [io::Create]).unwrap();
            for i in uints.iter() {
                file.write_le_u64(*i);
            }
        }

        // then read them back and check that they are the same
        {
            let file = io::file_reader(&path).unwrap();
            for i in uints.iter() {
                assert_eq!(file.read_le_u64(), *i);
            }
        }
    }

    #[test]
    fn test_read_write_be() {
        let path = Path("tmp/lib-io-test-read-write-be.tmp");
        let uints = [0, 1, 2, 42, 10_123, 100_123_456, u64::max_value];

        // write the ints to the file
        {
            let file = io::file_writer(&path, [io::Create]).unwrap();
            for i in uints.iter() {
                file.write_be_u64(*i);
            }
        }

        // then read them back and check that they are the same
        {
            let file = io::file_reader(&path).unwrap();
            for i in uints.iter() {
                assert_eq!(file.read_be_u64(), *i);
            }
        }
    }

    #[test]
    fn test_read_be_int_n() {
        let path = Path("tmp/lib-io-test-read-be-int-n.tmp");
        let ints = [i32::min_value, -123456, -42, -5, 0, 1, i32::max_value];

        // write the ints to the file
        {
            let file = io::file_writer(&path, [io::Create]).unwrap();
            for i in ints.iter() {
                file.write_be_i32(*i);
            }
        }

        // then read them back and check that they are the same
        {
            let file = io::file_reader(&path).unwrap();
            for i in ints.iter() {
                // this tests that the sign extension is working
                // (comparing the values as i32 would not test this)
                assert_eq!(file.read_be_int_n(4), *i as i64);
            }
        }
    }

    #[test]
    fn test_read_f32() {
        let path = Path("tmp/lib-io-test-read-f32.tmp");
        //big-endian floating-point 8.1250
        let buf = ~[0x41, 0x02, 0x00, 0x00];

        {
            let file = io::file_writer(&path, [io::Create]).unwrap();
            file.write(buf);
        }

        {
            let file = io::file_reader(&path).unwrap();
            let f = file.read_be_f32();
            assert_eq!(f, 8.1250);
        }
    }

    #[test]
    fn test_read_write_f32() {
        let path = Path("tmp/lib-io-test-read-write-f32.tmp");
        let f:f32 = 8.1250;

        {
            let file = io::file_writer(&path, [io::Create]).unwrap();
            file.write_be_f32(f);
            file.write_le_f32(f);
        }

        {
            let file = io::file_reader(&path).unwrap();
            assert_eq!(file.read_be_f32(), 8.1250);
            assert_eq!(file.read_le_f32(), 8.1250);
        }
    }
}
