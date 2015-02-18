// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Simple [DEFLATE][def]-based compression. This is a wrapper around the
//! [`miniz`][mz] library, which is a one-file pure-C implementation of zlib.
//!
//! [def]: https://en.wikipedia.org/wiki/DEFLATE
//! [mz]: https://code.google.com/p/miniz/

#![crate_name = "flate"]
#![unstable(feature = "rustc_private")]
#![staged_api]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/nightly/")]

#![feature(core)]
#![feature(int_uint)]
#![feature(libc)]
#![feature(staged_api)]

#[cfg(test)] #[macro_use] extern crate log;

extern crate libc;

use libc::{c_void, size_t, c_int};
use std::ops::Deref;
use std::ptr::Unique;
use std::slice;

pub struct Bytes {
    ptr: Unique<u8>,
    len: uint,
}

impl Deref for Bytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(*self.ptr, self.len) }
    }
}

impl Drop for Bytes {
    fn drop(&mut self) {
        unsafe { libc::free(*self.ptr as *mut _); }
    }
}

#[link(name = "miniz", kind = "static")]
extern {
    /// Raw miniz compression function.
    fn tdefl_compress_mem_to_heap(psrc_buf: *const c_void,
                                  src_buf_len: size_t,
                                  pout_len: *mut size_t,
                                  flags: c_int)
                                  -> *mut c_void;

    /// Raw miniz decompression function.
    fn tinfl_decompress_mem_to_heap(psrc_buf: *const c_void,
                                    src_buf_len: size_t,
                                    pout_len: *mut size_t,
                                    flags: c_int)
                                    -> *mut c_void;
}

static LZ_NORM : c_int = 0x80;  // LZ with 128 probes, "normal"
static TINFL_FLAG_PARSE_ZLIB_HEADER : c_int = 0x1; // parse zlib header and adler32 checksum
static TDEFL_WRITE_ZLIB_HEADER : c_int = 0x01000; // write zlib header and adler32 checksum

fn deflate_bytes_internal(bytes: &[u8], flags: c_int) -> Option<Bytes> {
    unsafe {
        let mut outsz : size_t = 0;
        let res = tdefl_compress_mem_to_heap(bytes.as_ptr() as *const _,
                                             bytes.len() as size_t,
                                             &mut outsz,
                                             flags);
        if !res.is_null() {
            let res = Unique::new(res as *mut u8);
            Some(Bytes { ptr: res, len: outsz as uint })
        } else {
            None
        }
    }
}

/// Compress a buffer, without writing any sort of header on the output.
pub fn deflate_bytes(bytes: &[u8]) -> Option<Bytes> {
    deflate_bytes_internal(bytes, LZ_NORM)
}

/// Compress a buffer, using a header that zlib can understand.
pub fn deflate_bytes_zlib(bytes: &[u8]) -> Option<Bytes> {
    deflate_bytes_internal(bytes, LZ_NORM | TDEFL_WRITE_ZLIB_HEADER)
}

fn inflate_bytes_internal(bytes: &[u8], flags: c_int) -> Option<Bytes> {
    unsafe {
        let mut outsz : size_t = 0;
        let res = tinfl_decompress_mem_to_heap(bytes.as_ptr() as *const _,
                                               bytes.len() as size_t,
                                               &mut outsz,
                                               flags);
        if !res.is_null() {
            let res = Unique::new(res as *mut u8);
            Some(Bytes { ptr: res, len: outsz as uint })
        } else {
            None
        }
    }
}

/// Decompress a buffer, without parsing any sort of header on the input.
pub fn inflate_bytes(bytes: &[u8]) -> Option<Bytes> {
    inflate_bytes_internal(bytes, 0)
}

/// Decompress a buffer that starts with a zlib header.
pub fn inflate_bytes_zlib(bytes: &[u8]) -> Option<Bytes> {
    inflate_bytes_internal(bytes, TINFL_FLAG_PARSE_ZLIB_HEADER)
}

#[cfg(test)]
mod tests {
    #![allow(deprecated)]
    use super::{inflate_bytes, deflate_bytes};
    use std::rand;
    use std::rand::Rng;

    #[test]
    fn test_flate_round_trip() {
        let mut r = rand::thread_rng();
        let mut words = vec!();
        for _ in 0..20 {
            let range = r.gen_range(1, 10);
            let v = r.gen_iter::<u8>().take(range).collect::<Vec<u8>>();
            words.push(v);
        }
        for _ in 0..20 {
            let mut input = vec![];
            for _ in 0..2000 {
                input.push_all(r.choose(&words).unwrap());
            }
            debug!("de/inflate of {} bytes of random word-sequences",
                   input.len());
            let cmp = deflate_bytes(&input).expect("deflation failed");
            let out = inflate_bytes(&cmp).expect("inflation failed");
            debug!("{} bytes deflated to {} ({:.1}% size)",
                   input.len(), cmp.len(),
                   100.0 * ((cmp.len() as f64) / (input.len() as f64)));
            assert_eq!(&*input, &*out);
        }
    }

    #[test]
    fn test_zlib_flate() {
        let bytes = vec!(1, 2, 3, 4, 5);
        let deflated = deflate_bytes(&bytes).expect("deflation failed");
        let inflated = inflate_bytes(&deflated).expect("inflation failed");
        assert_eq!(&*inflated, &*bytes);
    }
}
