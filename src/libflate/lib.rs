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
#![unstable(feature = "rustc_private", issue = "27812")]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/",
       test(attr(deny(warnings))))]
#![deny(warnings)]

#![feature(libc)]
#![feature(staged_api)]
#![feature(unique)]
#![cfg_attr(test, feature(rand))]

extern crate libc;

use libc::{c_int, c_void, size_t};
use std::fmt;
use std::ops::Deref;
use std::ptr::Unique;
use std::slice;

#[derive(Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Error {
    _unused: (),
}

impl Error {
    fn new() -> Error {
        Error { _unused: () }
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        "decompression error".fmt(f)
    }
}

pub struct Bytes {
    ptr: Unique<u8>,
    len: usize,
}

impl Deref for Bytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(*self.ptr, self.len) }
    }
}

impl Drop for Bytes {
    fn drop(&mut self) {
        unsafe {
            libc::free(*self.ptr as *mut _);
        }
    }
}

#[link(name = "miniz", kind = "static")]
#[cfg(not(cargobuild))]
extern "C" {}

extern "C" {
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

const LZ_FAST: c_int = 0x01;  // LZ with 1 probe, "fast"
const TDEFL_GREEDY_PARSING_FLAG: c_int = 0x04000; // fast greedy parsing instead of lazy parsing

/// Compress a buffer without writing any sort of header on the output. Fast
/// compression is used because it is almost twice as fast as default
/// compression and the compression ratio is only marginally worse.
pub fn deflate_bytes(bytes: &[u8]) -> Bytes {
    let flags = LZ_FAST | TDEFL_GREEDY_PARSING_FLAG;
    unsafe {
        let mut outsz: size_t = 0;
        let res = tdefl_compress_mem_to_heap(bytes.as_ptr() as *const _,
                                             bytes.len() as size_t,
                                             &mut outsz,
                                             flags);
        assert!(!res.is_null());
        Bytes {
            ptr: Unique::new(res as *mut u8),
            len: outsz as usize,
        }
    }
}

/// Decompress a buffer without parsing any sort of header on the input.
pub fn inflate_bytes(bytes: &[u8]) -> Result<Bytes, Error> {
    let flags = 0;
    unsafe {
        let mut outsz: size_t = 0;
        let res = tinfl_decompress_mem_to_heap(bytes.as_ptr() as *const _,
                                               bytes.len() as size_t,
                                               &mut outsz,
                                               flags);
        if !res.is_null() {
            Ok(Bytes {
                ptr: Unique::new(res as *mut u8),
                len: outsz as usize,
            })
        } else {
            Err(Error::new())
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(deprecated)]
    use super::{deflate_bytes, inflate_bytes};
    use std::__rand::{Rng, thread_rng};

    #[test]
    fn test_flate_round_trip() {
        let mut r = thread_rng();
        let mut words = vec![];
        for _ in 0..20 {
            let range = r.gen_range(1, 10);
            let v = r.gen_iter::<u8>().take(range).collect::<Vec<u8>>();
            words.push(v);
        }
        for _ in 0..20 {
            let mut input = vec![];
            for _ in 0..2000 {
                input.extend_from_slice(r.choose(&words).unwrap());
            }
            let cmp = deflate_bytes(&input);
            let out = inflate_bytes(&cmp).unwrap();
            assert_eq!(&*input, &*out);
        }
    }

    #[test]
    fn test_zlib_flate() {
        let bytes = vec![1, 2, 3, 4, 5];
        let deflated = deflate_bytes(&bytes);
        let inflated = inflate_bytes(&deflated).unwrap();
        assert_eq!(&*inflated, &*bytes);
    }
}
