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

Simple compression using the very fast LZ4 algorithm

*/

#[allow(missing_doc)];

use core::prelude::*;

use core::libc::{c_void, c_int};
use core::vec;
use core::num;
use core::io;
use core::sys;
use core::vec::raw::to_mut_ptr;

priv mod rustrt {
    use core::libc::{c_int, c_void};

    #[link_name = "rustrt"]
    pub extern {
        unsafe fn LZ4_compress(source: *const c_void, dest: *mut c_void, inputSize: c_int) -> c_int;
        unsafe fn LZ4_compressHC(source: *const c_void, dest: *mut c_void, inputSize: c_int)
            -> c_int;
        unsafe fn LZ4_decompress_safe(source: *const c_void, dest: *mut c_void, inputSize: c_int,
                                      maxOutputSize: c_int) -> c_int;
    }
}


/// Worst case compressed size
pub fn LZ4_compressBound(size: uint) -> uint { size + size/255 + 16 }

/// Container for LZ4-compressed data, because LZ4 doesn't define its own container
pub struct LZ4Container {
    uncompressed_size: uint,
    buf_size: uint,
    buf: ~[u8]
}

/// Compress a buffer
pub fn compress_bytes(bytes: &const [u8], high_compression: bool) -> LZ4Container {
    let max_cint: c_int = num::Bounded::max_value();
    assert!(bytes.len() <= max_cint as uint, "buffer too long");
    let mut buf: ~[u8] = vec::with_capacity(LZ4_compressBound(bytes.len()));
    let mut res = 0;
    do vec::as_const_buf(bytes) |b, len| {
        unsafe {
            if !high_compression {
                res = rustrt::LZ4_compress(b as *c_void, to_mut_ptr(buf) as *mut c_void,
                                           len as c_int);
            } else {
                res = rustrt::LZ4_compressHC(b as *c_void, to_mut_ptr(buf) as *mut c_void,
                                             len as c_int);
            }
            vec::raw::set_len(&mut buf, res as uint);
            assert!(res as int != 0, "LZ4_compress(HC) failed");
        }
    }
    // FIXME #4960: realloc buffer to res bytes
    return LZ4Container{ uncompressed_size: bytes.len(), buf_size: buf.len(), buf: buf }
}

impl LZ4Container {
    /// Decompress LZ4 data. Returns None if the input buffer was malformed or didn't decompress
    /// to `size` bytes.
    pub fn decompress(&self) -> Option<~[u8]> {
        let mut ret: Option<~[u8]> = None;
        do vec::as_const_buf(self.buf) |b, len| {
            let mut out: ~[u8] = vec::with_capacity(self.uncompressed_size as uint);
            unsafe {
                let res = rustrt::LZ4_decompress_safe(b as *c_void, to_mut_ptr(out) as *mut c_void,
                                                      len as c_int, self.uncompressed_size as c_int);
                if res != self.uncompressed_size as c_int {
                    warn!("LZ4_decompress_safe returned %?", res);
                    ret = None
                } else {
                    vec::raw::set_len(&mut out, res as uint);
                    ret = Some(out)
                }
            }
        }
        ret
    }
    /// Create an LZ4Container out of bytes
    pub fn from_bytes(bytes: &[u8]) -> LZ4Container {
        do io::with_bytes_reader(bytes) |rdr| {
            let uncompressed_size = rdr.read_le_uint();
            let buf_size = rdr.read_le_uint();
            let remaining = bytes.len() - rdr.tell();
            assert!(remaining >= buf_size,
                    fmt!("header wants more bytes than present in buffer (wanted %?, found %?)",
                         buf_size, remaining));
            let buf = bytes.slice(rdr.tell(), rdr.tell() + buf_size).to_owned();
            assert_eq!(buf_size, buf.len());
            LZ4Container { uncompressed_size: uncompressed_size, buf_size: buf_size, buf: buf }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::rand;
    use core::rand::RngUtil;

    #[test]
    #[allow(non_implicitly_copyable_typarams)]
    fn test_round_trip() {
        let mut r = rand::rng();
        let mut words = ~[];
        for 20.times {
            let range = r.gen_uint_range(1, 10);
            words.push(r.gen_bytes(range));
        }
        for 20.times {
            let mut in = ~[];
            for 2000.times {
                in.push_all(r.choose(words));
            }
            debug!("de/inflate of %u bytes of random word-sequences",
                   in.len());
            let cmp = compress_bytes(in, true);
            debug!("compressed size reported as %?", cmp.size);
            let out = cmp.decompress().unwrap();
            debug!("%u bytes compressed to %u (%.1f%% size) and was decompressed to %?",
                   in.len(), cmp.buf.len(),
                   100.0 * ((cmp.buf.len() as float) / (in.len() as float)), out.len());
            assert_eq!(in, out);
        }
    }
}
