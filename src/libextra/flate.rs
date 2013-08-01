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

Simple compression

*/

#[allow(missing_doc)];

use std::libc::{c_void, size_t, c_int};
use std::libc;
use std::vec;

pub mod rustrt {
    use std::libc::{c_int, c_void, size_t};

    #[link_name = "rustrt"]
    extern {
        pub unsafe fn tdefl_compress_mem_to_heap(psrc_buf: *const c_void,
                                                 src_buf_len: size_t,
                                                 pout_len: *mut size_t,
                                                 flags: c_int)
                                                 -> *c_void;

        pub unsafe fn tinfl_decompress_mem_to_heap(psrc_buf: *const c_void,
                                                   src_buf_len: size_t,
                                                   pout_len: *mut size_t,
                                                   flags: c_int)
                                                   -> *c_void;
    }
}

static LZ_NONE : c_int = 0x0;   // Huffman-coding only.
static LZ_FAST : c_int = 0x1;   // LZ with only one probe
static LZ_NORM : c_int = 0x80;  // LZ with 128 probes, "normal"
static LZ_BEST : c_int = 0xfff; // LZ with 4095 probes, "best"

pub fn deflate_bytes(bytes: &[u8]) -> ~[u8] {
    do bytes.as_imm_buf |b, len| {
        unsafe {
            let mut outsz : size_t = 0;
            let res =
                rustrt::tdefl_compress_mem_to_heap(b as *c_void,
                                                   len as size_t,
                                                   &mut outsz,
                                                   LZ_NORM);
            assert!(res as int != 0);
            let out = vec::raw::from_buf_raw(res as *u8,
                                             outsz as uint);
            libc::free(res);
            out
        }
    }
}

pub fn inflate_bytes(bytes: &[u8]) -> ~[u8] {
    do bytes.as_imm_buf |b, len| {
        unsafe {
            let mut outsz : size_t = 0;
            let res =
                rustrt::tinfl_decompress_mem_to_heap(b as *c_void,
                                                     len as size_t,
                                                     &mut outsz,
                                                     0);
            assert!(res as int != 0);
            let out = vec::raw::from_buf_raw(res as *u8,
                                            outsz as uint);
            libc::free(res);
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rand;
    use std::rand::RngUtil;

    #[test]
    fn test_flate_round_trip() {
        let mut r = rand::rng();
        let mut words = ~[];
        do 20.times {
            let range = r.gen_uint_range(1, 10);
            words.push(r.gen_bytes(range));
        }
        do 20.times {
            let mut input = ~[];
            do 2000.times {
                input.push_all(r.choose(words));
            }
            debug!("de/inflate of %u bytes of random word-sequences",
                   input.len());
            let cmp = deflate_bytes(input);
            let out = inflate_bytes(cmp);
            debug!("%u bytes deflated to %u (%.1f%% size)",
                   input.len(), cmp.len(),
                   100.0 * ((cmp.len() as float) / (input.len() as float)));
            assert_eq!(input, out);
        }
    }
}
