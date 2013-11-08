// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0.  If a copy of the MPL was not distributed with this file,
// You can obtain one at http://mozilla.org/MPL/2.0/.
// 
// Software distributed under the License is distributed on an "AS IS" basis,
// WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License for 
// the specific language governing rights and limitations under the License.
//
// The Original Code is: deflate.rs
// The Initial Developer of the Original Code is: William Wong (williamw520@gmail.com)
// Portions created by William Wong are Copyright (C) 2013 William Wong, All Rights Reserved.



/*!

The deflate module provides compression and decompression support with the
DEFLATE algorithm.  Its wraps the miniz.c as basis for the implementation
for DEFLATE and provides an easy to use API in Rust.

There are two sets of API: cursor-style API for operating data in batches,
and the stream-style API for compressing and decompressing the whole streams.

The compress_read and decompress_read API support reading and writing batches
of data.  These are caller-driven functions where the caller is responsible
to call them repeatedly to process all data.

The compress_stream and decompress_stream are the stream-style API that take
a source read_fn and a destination write_fn to compress/decompress all
the data from the source to the destination stream automatically.  It is
callee-driven with an internal loop running until all the data have been
processed.  It's more efficient with less buffer copying.

Note that often time a DEFLATE data stream is embedded in an outer containing
stream, e.g. a zip file or a gzip file.  The API here only work with the DEFLATE
portion of the data stream.  Be careful not to pass in data belonging to the
outer stream.  The only exception is during decompression.  Since the total 
length of the compressed data is sometime unknown during decompression (in 
streaming situation), the decompression API will read as much data as possible
from read_fn to attempt to decompress.  Any extra unprocessed data that are read
in after the compressed data will be passed back to the caller.  The caller
should treat the extra data as part of the outer containing stream.

See gzip.rs for sample usage.

*/

use std::rt::io::{Reader, Writer};
use std::{vec, num, ptr};
use std::libc::{c_void, size_t, c_int, c_uint};



/// Deflate function return status
pub enum DeflateStatus {
    /// Invalid parameters passed into API. e.g. buffer too small
    DeflateStatusBadParam = -2,
    /// The callback write fn has failed
    DeflateStatusPutBufFailed = -1,
    /// Intermediate compression call succeeded but expecting more data to compress.
    DeflateStatusOkay = 0,
    /// All data have been compressed and finalized.
    DeflateStatusDone = 1,
    /// The callback write fn wants to abort the compression operation.  Stream-loop will be broken and returned.
    DeflateStatusAbort = -9998,
    /// Unknown status from low layer
    DeflateStatusUnknown = -9999,
}

impl DeflateStatus {
    fn from_status(status: c_int) -> DeflateStatus {
        match status {
            -2 => DeflateStatusBadParam,
            -1 => DeflateStatusPutBufFailed,
            0  => DeflateStatusOkay,
            1  => DeflateStatusDone,
            _  => DeflateStatusUnknown
        }
    }
}

/// INFLATE function return status
pub enum InflateStatus {
    /// The inflator needs 1 or more input bytes to make forward progress, but the caller is indicating that no more are available. The compressed data is probably corrupted.
    InflateStatusFailedCannotMakeProgress = -4,
    /// Invalid parameters passed into API. e.g. buffer too small.
    InflateStatusBadParam = -3,
    /// The inflator is finished but the ADLER32 check of the uncompressed data didn't match.  Calling it again it'll return InflateStatusDone.
    StatusAdler32Mismatch = -2,
    /// The inflator has failed (corrupted input, bad buffer, etc.).
    InflateStatusFailed = -1,
    /// The inflator has returned every byte of uncompressed data that it can, has consumed every byte that it needed, has successfully reached the end of the deflate stream.
    InflateStatusDone = 0,
    /// Need more input data before the inflator can make any more progress.  Can supply more data or set the final_input flag in call.
    /// If the source data was corrupted, it's possible but unlikely for the inflator to keep on demanding input to proceed.
    InflateStatusNeedsMoreInput = 1,
    /// The output buffer is full, and the inflator has more bytes of uncompressed data to process but it cannot write to the output buffer.
    /// Uncompressing is paused until room are made in the output buffer.
    InflateStatusHasMoreOutput = 2,
    /// The write_fn has returned a flag to abort the uncompression process.
    InflateStatusAbort = -9998,
    /// Unknown status from low layer.
    InflateStatusUnknown = -9999,
}

impl InflateStatus {
    fn from_status(status: c_int) -> InflateStatus {
        match status {
            -4 => InflateStatusFailedCannotMakeProgress,
            -3 => InflateStatusBadParam,
            -2 => StatusAdler32Mismatch,
            -1 => InflateStatusFailed,
            0  => InflateStatusDone,
            1  => InflateStatusNeedsMoreInput,
            2  => InflateStatusHasMoreOutput,
            _  => InflateStatusUnknown
        }
    }
}


/// The number of dictionary probes to use at each compression level (0-9). 0=implies fastest/minimal possible probing, 9=best compression but slowest.
pub static MAX_COMPRESS_LEVEL : uint = 9;
static TDEFL_NUM_PROBES : [c_uint, ..10] = [ 0 as c_uint, 2, 8, 32, 128, 256, 512, 1024, 2048, 4095 ];

/// Max size of the LZ dictionary is 32K at the beginning of an out_buf, which becomes the minimum output buffer size for decompression.
pub static MIN_DECOMPRESS_BUF_SIZE : uint = 32768;

/// The buf_size_factor for internal IO buffers.
pub static MIN_SIZE_FACTOR : uint = 5;              // minimum size factor: 2^5 * 1K = 32K
pub static DEFAULT_SIZE_FACTOR : uint = 8;          // default size factor: 2^8 * 1K = 256K

// Define the Miniz flags here for internal use
static TDEFL_WRITE_ZLIB_HEADER : c_uint             = 0x01000;
static TDEFL_COMPUTE_ADLER32 : c_uint               = 0x02000;
static TDEFL_GREEDY_PARSING_FLAG : c_uint           = 0x04000;
static TDEFL_NONDETERMINISTIC_PARSING_FLAG : c_uint = 0x08000;
static TDEFL_RLE_MATCHES : c_uint                   = 0x10000;
static TDEFL_FILTER_MATCHES : c_uint                = 0x20000;
static TDEFL_FORCE_ALL_STATIC_BLOCKS : c_uint       = 0x40000;
static TDEFL_FORCE_ALL_RAW_BLOCKS : c_uint          = 0x08000;

static TDEFL_NO_FLUSH : c_int   = 0;
static TDEFL_SYNC_FLUSH : c_int = 2;
static TDEFL_FULL_FLUSH : c_int = 3;
static TDEFL_FINISH : c_int     = 4;

static TINFL_FLAG_PARSE_ZLIB_HEADER : c_uint                = 1;
static TINFL_FLAG_HAS_MORE_INPUT : c_uint                   = 2;
static TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF : c_uint    = 4;
static TINFL_FLAG_COMPUTE_ADLER32 : c_uint                  = 8;

static LZ_NONE : c_int = 0x0;   // Huffman-coding only.
static LZ_FAST : c_int = 0x1;   // LZ with only one probe
static LZ_NORM : c_int = 0x80;  // LZ with 128 probes, "normal"
static LZ_BEST : c_int = 0xfff; // LZ with 4095 probes, "best"


mod rustrt {
    use std::libc::{c_void, size_t, c_int, c_uint};

    #[link_name = "rustrt"]
    extern {
        // The DEFLATE algorithm is handled by the Miniz package in C/C++ land.
        // Define the API into miniz.cpp.
        pub fn tdefl_compressor_alloc() -> *c_void;
        pub fn tdefl_compressor_free(pDeflator: *c_void);
        pub fn tdefl_init(tdefl_compressor: *c_void, 
                          pPut_buf_func: *c_void, 
                          pPut_buf_user: *c_void, 
                          compress_flags: c_int) -> c_int;
        pub fn tdefl_compress(tdefl_compressor: *c_void, 
                              pIn_buf: *c_void, 
                              pIn_buf_size: *mut size_t, 
                              pOut_buf: *c_void, 
                              pOut_buf_size: *mut size_t, 
                              tdefl_flush: c_int) -> c_int;

        pub fn tinfl_decompressor_alloc() -> *c_void;
        pub fn tinfl_decompressor_free(tinfl_decompressor: *c_void);
        pub fn tinfl_decompress(tinfl_decompressor: *c_void, 
                                pIn_buf_next: *c_void, 
                                pIn_buf_size: *mut size_t, 
                                pOut_buf_start: *c_void, 
                                pOut_buf_next: *c_void, 
                                pOut_buf_size: *mut size_t, 
                                decompress_flags: c_uint) -> c_int;

        pub fn tdefl_compress_mem_to_heap(psrc_buf: *c_void,
                                          src_buf_len: size_t,
                                          pout_len: *mut size_t,
                                          flags: c_int)
                                          -> *c_void;
        pub fn tinfl_decompress_mem_to_heap(psrc_buf: *c_void,
                                            src_buf_len: size_t,
                                            pout_len: *mut size_t,
                                            flags: c_int)
                                            -> *c_void;

        pub fn mz_free(pBuf: *c_void) -> *c_void;
    }
}


/// Calculate the IO buffer size in bytes given a buf_size_factor.
/// buf_size_factor is a power of 2.   buf_in_bytes = 1024 * 2 ^ buf_size_factor
pub fn calc_buf_size(buf_size_factor: uint) -> uint {
    return 1024 * num::pow_with_uint(2, buf_size_factor);
}


/// Compression data structure
struct Deflator {
    tdefl_compressor: *c_void,
    in_buf: ~[u8],
    out_buf: ~[u8],
    in_offset: uint,
    in_buf_total: uint,
    out_offset: uint,
    read_total: uint,
    write_total: uint,
}

impl Deflator {
    /// Creates the Deflator structure and allocates the underlying tdefl_compressor structure.
    pub fn new() -> Deflator {
        Deflator::with_size_factor(DEFAULT_SIZE_FACTOR)
    }

    /// Creates the Deflator structure and allocates the underlying tdefl_compressor structure.
    /// Allocates the IO buffers with buf_size_factor.  The buf_size_factor is a power of 2 of K: 2^buf_size_factor X 1K.
    pub fn with_size_factor(buf_size_factor: uint) -> Deflator {
        #[fixed_stack_segment];
        #[inline(never)];
        unsafe {
            Deflator {
                tdefl_compressor:   rustrt::tdefl_compressor_alloc(),
                in_buf:             vec::from_elem(calc_buf_size(buf_size_factor), 0u8),
                out_buf:            vec::from_elem(calc_buf_size(buf_size_factor) + MIN_DECOMPRESS_BUF_SIZE, 0u8),
                in_offset:          0u,
                in_buf_total:       0u,
                out_offset:         0u,
                read_total:         0u,
                write_total:        0u,
            }
        }
    }

    /// Releases the underlying tdefl_compressor structure.  After this call, the instance can not be used any more.
    /// Called by the drop() destructor.
    pub fn free(&mut self) {
        #[fixed_stack_segment];
        #[inline(never)];
        unsafe {
            if self.tdefl_compressor != ptr::null() {
                rustrt::tdefl_compressor_free(self.tdefl_compressor);
            }
            self.tdefl_compressor = ptr::null();
        }
    }

    /// Initializes the Deflator.
    ///
    /// compress_level is 0 to 10, where 0 is the fastest with decompressed raw data and 9 is the slowest with best compression.
    /// add_zlib_header set to true to add the ZLib-format header in front of and an ADLER32 CRC at the end of the deflated data.
    /// add_crc32 set to true to add an ADLER32 CRC at the end of the deflated data regardless how add_zlib is set.
    pub fn init(&self, compress_level: uint, add_zlib_header: bool, add_crc32: bool) -> DeflateStatus {
        #[fixed_stack_segment];
        #[inline(never)];

        let compress_level = num::min(MAX_COMPRESS_LEVEL, compress_level);
        let compress_flags = 
            TDEFL_NUM_PROBES[compress_level] | 
            (if compress_level <= 3 { TDEFL_GREEDY_PARSING_FLAG } else { 0 }) |
            (if compress_level > 0  { 0 } else { TDEFL_FORCE_ALL_RAW_BLOCKS }) |
            (if add_zlib_header { TDEFL_WRITE_ZLIB_HEADER } else { 0 }) |
            (if add_crc32 { TDEFL_COMPUTE_ADLER32 } else { 0 });

        unsafe {
            let status = rustrt::tdefl_init(self.tdefl_compressor, ptr::null(), ptr::null(), compress_flags as c_int);
            return DeflateStatus::from_status(status);
        }
    }

    /// Compresses all data read from the reader and writes the compressed data to the writer.
    /// Runs until reading EOF from reader.  Waits on read or wait on write if they are blocked.
    /// Demo usage of compress_stream().
    pub fn compress_stream_rw<R: Reader, W: Writer>(&mut self, in_reader: &mut R, out_writer: &mut W) -> DeflateStatus {
        self.compress_stream(
            // upcall function to read data for compression
            |in_buf| {
                if in_reader.eof() {
                    0                           // Return 0 for EOF
                } else {
                    match in_reader.read(in_buf) {
                        Some(nread) => nread,   // Return number of bytes read, including 0 for EOF
                        None => 0               // Return 0 for EOF
                    }
                }
            },
            // upcall function to write compressed data
            |out_buf, is_eof| {
                out_writer.write(out_buf);
                if is_eof {
                    out_writer.flush();
                }
                false                           // don't abort
            })
    }

    /// Compresses using callback functions to caller (upcalls) to read data and write data.
    /// The input to compress are supplied by the read_fn callback function of the caller.
    /// The compressed data are sent to the write_fn callback function of the caller.
    /// Runs until reading EOF from read_fn.  Waits on read or wait on write if they are blocked.
    ///
    /// The callback read_fn returns one batch of input data at a time in the in_buf buffer.
    /// It returns the number of bytes read.  Returns 0 for EOF or no more data.
    ///
    /// The callback write_fn function takes in the out_buf buffer containing one batch of compressed data.
    /// The is_eof flag is set for the last batch of compressed data.
    /// Write_fn can return an abort flag to abort the compression.
    pub fn compress_stream(&mut self, 
                           read_fn:  &fn(in_buf: &mut [u8])->uint, 
                           write_fn: &fn(out_buf: &[u8], is_eof: bool)->bool) -> DeflateStatus {

        let out_buf_total = self.out_buf.len();

        loop {
            // Read some input data if in_buf is empty
            if self.in_offset == self.in_buf_total {
                self.in_buf_total = read_fn(self.in_buf);               // in_buf_total == 0 for EOF
                self.in_offset = 0;
            }

            let mut in_bytes = self.in_buf_total - self.in_offset;      // number of bytes to compress in this batch;
            let mut out_bytes = out_buf_total - self.out_offset;        // number of bytes of space avaiable in the out_buf;
            let final_input = self.in_buf_total == 0;
            let status = self.compress_buf(self.in_buf, self.in_offset, &mut in_bytes, self.out_buf, self.out_offset, &mut out_bytes, final_input);

            self.in_offset += in_bytes;                                 // advance offset by the number of bytes consumed.
            self.out_offset += out_bytes;                               // advance offset by the number of bytes written.

            match status {
                DeflateStatusOkay => {
                    // If out_buf is full, write its content out.  Reset it.
                    if self.out_offset == out_buf_total {
                        if write_fn(self.out_buf, false) {
                            return DeflateStatusAbort;
                        }
                        self.out_offset = 0;
                    }
                },
                DeflateStatusDone => {
                    // Write the remaining content in out_buf out.
                    write_fn(self.out_buf.slice(0, self.out_offset), true);
                    return DeflateStatusDone;
                },
                _ => return status  // Return error
            }
        }
    }

    /// Compresses one batch of input data at a time.  Caller drives the write loop.
    /// Caller needs to call this function repeatedly with new data until all data are written out.
    //  This approach has more buffer copying than the stream approach.
    ///
    /// Input to compress is supplied in input_buf, which will be fully compressed and written out before returning.
    /// The compressed data are sent to caller via the write_fn callback.
    /// The final_write flag must be set for the last batch of data to compress, to finalize the compressed data.
    /// The last batch of data can be zero-length.
    pub fn compress_write(&mut self,
                          input_buf: &[u8],
                          final_write: bool,
                          write_fn: &fn(out_buf: &[u8], is_eof: bool)) -> DeflateStatus {

        let out_buf_total = self.out_buf.len();
        let input_total = input_buf.len();
        let mut input_offset = 0;
        let mut input_remaining;

        loop {
            // Move some input data into in_buf if in_buf is empty
            if self.in_offset == self.in_buf_total {
                let copy_len = num::min(self.in_buf.len(), input_total - input_offset);
                vec::bytes::copy_memory(self.in_buf, input_buf.slice(input_offset, input_offset + copy_len), copy_len);
                input_offset += copy_len;
                self.in_offset = 0;
                self.in_buf_total = copy_len;
                self.read_total += copy_len;
            }
            input_remaining = input_total - input_offset;

            let mut in_bytes = self.in_buf_total - self.in_offset;      // number of bytes to compress in this batch
            let mut out_bytes = out_buf_total - self.out_offset;        // number of bytes of space avaiable in the out_buf
            let final_input = (final_write && input_remaining == 0);    // final_write is set and last batch in input_buf
            let status = self.compress_buf(self.in_buf, self.in_offset, &mut in_bytes, 
                                           self.out_buf, self.out_offset, &mut out_bytes, final_input);
            self.in_offset += in_bytes;                                 // advance offset by the number of bytes consumed
            self.out_offset += out_bytes;                               // advance offset by the number of bytes written

            match status {
                DeflateStatusOkay => {
                    // Only when out_buf is full, write its content out.  Reset it.
                    if self.out_offset == out_buf_total {
                        write_fn(self.out_buf, false);
                        self.write_total += self.out_offset;
                        self.out_offset = 0;
                    }
                },
                DeflateStatusDone => {
                    // Write the remaining content in out_buf out.
                    write_fn(self.out_buf.slice(0, self.out_offset), true);
                    self.write_total += self.out_offset;
                    return DeflateStatusDone;
                },
                _ => return status  // Return error
            }

            // Has compressed all input_buf for the current non-final write.  Return to caller.
            if !final_write && input_remaining == 0 {
                return DeflateStatusOkay;
            }
            // Important: for the final_write, need to loop to flush all remaining data in 
            // in_buf and out_buf until DeflateStatusDone.
        }
    }

    /// Low level compress method to compress input data to DEFLATE compliant compressed data.
    /// You really need to know what you are doing to call this directly.  It's fragile with edge cases.
    /// It has multiple modes of operation depending on the parameters.
    ///
    /// in_buf has the input data to be compressed.
    /// in_offset is the offset into in_buf to start reading the data.
    /// in_bytes is the number of bytes to read starting from in_offset, as call input.
    /// in_bytes is the number of bytes has been consumed, as call output.
    /// out_buf is the compressed output data.  The size of out_buf must be as big or bigger than in_buf.
    /// out_offset is the offset into out_buf to start writing the compressed data.
    /// out_bytes is the number of bytes available to store the compressed data starting from out_offset, as call input.
    /// out_bytes is the number of bytes has been used up to store the compressed data, as call output.
    /// final_input set to false if there will be calls again for more input data, set to true for the last batch of input.
    pub fn compress_buf(&self, 
                        in_buf:  &[u8], in_offset:  uint, in_bytes:  &mut uint, 
                        out_buf: &[u8], out_offset: uint, out_bytes: &mut uint, 
                        final_input: bool) -> DeflateStatus {
        #[fixed_stack_segment];
        #[inline(never)];

        let mut status : c_int = 0;
        let mut in_bytes_sz  = *in_bytes as size_t;
        let mut out_bytes_sz = *out_bytes as size_t;
        let in_buf_next  = in_buf.slice(in_offset, in_offset + *in_bytes);
        let out_buf_next = out_buf.slice(out_offset, out_offset + *out_bytes);

        do in_buf_next.as_imm_buf |in_next_ptr, _| {
            do out_buf_next.as_imm_buf |out_next_ptr, _| {
                unsafe {
                    status = rustrt::tdefl_compress(self.tdefl_compressor, 
                                                    in_next_ptr as *c_void, 
                                                    &mut in_bytes_sz, 
                                                    out_next_ptr as *c_void, 
                                                    &mut out_bytes_sz, 
                                                    if final_input { TDEFL_FINISH } else { TDEFL_NO_FLUSH });
                }
            }
        }

        *in_bytes = in_bytes_sz as uint;
        *out_bytes = out_bytes_sz as uint;
        return DeflateStatus::from_status(status);
    }

}

/// destructor
impl Drop for Deflator {
    fn drop(&mut self) {
        self.free();
    }
}


/// Decompression data structure
struct Inflator {
    tinfl_decompressor: *c_void,
    in_buf: ~[u8],
    out_buf: ~[u8],
    in_offset: uint,                // beginning of the pending input data for decompression
    in_buf_total: uint,             // end of the pending input data for decompression
    out_begin: uint,                // beginning of cached output
    out_offset: uint,               // end of the cached output, beginning of available space for decompression.
    decomp_done: bool,
    read_total: uint,
    write_total: uint,
}

impl Inflator {
    /// Creates the Inflator structure and allocates the underlying tdefl_compressor structure.
    pub fn new() -> Inflator {
        Inflator::with_size_factor(DEFAULT_SIZE_FACTOR)
    }

    /// Creates the Inflator structure and allocates the underlying tdefl_compressor structure.
    /// Allocates the IO buffers with buf_size_factor.  The buf_size_factor is a power of 2 of K: 2^buf_size_factor X 1K.
    pub fn with_size_factor(buf_size_factor: uint) -> Inflator {
        #[fixed_stack_segment];
        #[inline(never)];
        unsafe {
            Inflator {
                tinfl_decompressor: rustrt::tinfl_decompressor_alloc(),
                in_buf:             vec::from_elem(calc_buf_size(buf_size_factor), 0u8),
                out_buf:            vec::from_elem(calc_buf_size(buf_size_factor) * 2, 0u8),  // out_buf size must be power of 2
                in_offset:          0u,
                in_buf_total:       0u,
                out_begin:          0u,
                out_offset:         0u,
                decomp_done:        false,
                read_total:         0u,
                write_total:        0u,
            }
        }
    }

    /// Releases the underlying tinfl_decompressor structure.  After this call, the instance must not be used anymore.
    pub fn free(&mut self) {
        #[fixed_stack_segment];
        #[inline(never)];
        unsafe {
            if self.tinfl_decompressor != ptr::null() {
                rustrt::tinfl_decompressor_free(self.tinfl_decompressor);
            }
            self.tinfl_decompressor = ptr::null();
        }
    }

    /// Reads the input data from reader, decompressed them, and writes them to writer.
    /// Any extra input data from the reader beyond the compressed data are discarded.
    /// Loops until reading EOF from reader.  Waits on read or wait on write if they are blocked.
    /// Demo usage of decompress_stream().
    pub fn decompress_stream_rw<R: Reader, W: Writer>(&mut self, in_reader: &mut R, out_writer: &mut W) -> InflateStatus {
        self.decompress_stream(
            // upcall function to read input data for decompression
            |in_buf| {
                if in_reader.eof() {
                    0                           // Return 0 for EOF
                } else {
                    match in_reader.read(in_buf) {
                        Some(nread) => nread,   // Return number of bytes read, including 0 for EOF
                        None => 0               // REturn 0 for EOF
                    }
                }
            },
            // upcall function to write the decompressed data
            |out_buf, is_eof| {
                out_writer.write(out_buf);
                if is_eof {                     // End of the decompressed data
                    out_writer.flush();
                }
                false                           // Don't abort
            },
            // upcall function to handle the remaining input data that are not part of the compressed data.
            |_ /*rest_buf*/| {
                // The extra data are discarded.
                //out_writer.write(rest_buf);
                //out_writer.flush();
            } )
    }

    /// Reads the input data from read_fn, decompresses them, and writes them to write_fn.
    /// This drives the read loop internally.  Loops until reading EOF from read_fn.  Less buffer copying.
    /// The input data are read as much as possible to process the compressed data.  There might be left-over
    /// data not part of compressed data.  The remaining unprocessed input data are sent back to caller via the rest_fn.
    ///
    /// The input data to decompress are supplied by the read_fn callback function of the caller.
    /// The decompressed data are sent to the write_fn callback function of the caller.
    /// The remaining unprocessed input data are sent back to the rest_fn callback function of the caller.
    ///
    /// The callback read_fn returns one batch of input data at a time in the in_buf buffer.
    /// It returns the number of bytes read.  Returns 0 for EOF or no more data.
    ///
    /// The callback write_fn takes in the out_buf buffer containing one batch of decompressed data.
    /// The is_eof flag is set for the last batch of decompressed data.
    /// Write_fn can return an abort flag to abort the decompression.
    ///
    /// The callback rest_fn takes in the rest_buf buffer containing any extra unprocessed input data.
    pub fn decompress_stream(&mut self, 
                             read_fn:  &fn(in_buf: &mut [u8])->uint, 
                             write_fn: &fn(out_buf: &[u8], is_eof: bool)->bool,
                             rest_fn:  &fn(rest_buf: &[u8]) ) -> InflateStatus {

        let out_buf_total = self.out_buf.len();

        loop {
            // Read some input data if in_buf is empty
            if self.in_offset == self.in_buf_total {
                self.in_offset = 0;
                self.in_buf_total = read_fn(self.in_buf);       // in_buf_total == 0 for EOF
                self.read_total += self.in_buf_total;
            }

            let mut in_bytes = self.in_buf_total - self.in_offset;
            let mut out_bytes = out_buf_total - self.out_offset;
            let final_input = self.in_buf_total == 0;
            let status = self.decompress_buf(self.in_buf, self.in_offset, &mut in_bytes, final_input, 
                                             self.out_buf, self.out_offset, &mut out_bytes, true);
            self.in_offset += in_bytes;
            self.out_offset += out_bytes;

            match status {
                InflateStatusNeedsMoreInput | InflateStatusHasMoreOutput => {
                    // The internal out_buf is full.  Time to writ it out.
                    // Important to process until out_buf is full because the LZ dictionary 
                    // at the beginning of the buffer is being re-used until buf is full.
                    if self.out_offset == out_buf_total {
                        self.write_total += self.out_offset;
                        if write_fn(self.out_buf, false) {
                            return InflateStatusAbort;
                        }
                        self.out_offset = 0;
                    }
                },
                InflateStatusDone => {
                    self.write_total += self.out_offset;
                    write_fn(self.out_buf.slice(0, self.out_offset), true);
                    rest_fn(self.in_buf.slice(self.in_offset, self.in_buf_total));
                    return status;
                },
                _ => return status  // return error
            }
        }
    }

    /// Decompresses one batch of input data at a time.  The decompressed data are returned in output_buf.
    /// The length of the output data is returned in Ok(output_len).
    /// Caller calls this function repeatedly to read all the decompressed data until output_len is 0.
    //  This approach has more buffer copying than the stream approach.
    /// The input to decompress are supplied by the read_fn callback function of the caller.
    /// The decompressed data are returned to caller one batch at a time.
    /// After reaching the end of output, the remaining unprocessed input data can be retrieved with get_rest().
    pub fn decompress_read(&mut self, 
                           read_fn:  &fn(in_buf: &mut [u8])->uint, 
                           output_buf: &mut [u8]) -> Result<uint, InflateStatus> {

        let out_buf_total = self.out_buf.len();

        loop {
            // Drain all output data from the internal out_buf.
            let out_available_bytes = self.out_offset - self.out_begin;
            if out_available_bytes > 0 {
                let output_len = num::min(output_buf.len(), out_available_bytes);
                vec::bytes::copy_memory(output_buf, self.out_buf.slice(self.out_begin, self.out_begin + output_len), output_len);
                self.out_begin += output_len;
                return Ok(output_len);
            }

            // If it's already done, just return
            if self.decomp_done {
                return Ok(0);
            }

            // Loop to decompress to fill up the internal out_buf
            self.out_begin = 0;
            self.out_offset = 0;
            loop {
                // Read some input data if in_buf is empty
                if self.in_offset == self.in_buf_total {
                    self.in_buf_total = read_fn(self.in_buf);       // in_buf_total == 0 for EOF
                    self.in_offset = 0;
                }

                let mut in_bytes = self.in_buf_total - self.in_offset;
                let mut out_bytes = out_buf_total - self.out_offset;
                let final_input = self.in_buf_total == 0;
                let status = self.decompress_buf(self.in_buf, self.in_offset, &mut in_bytes, final_input, 
                                                 self.out_buf, self.out_offset, &mut out_bytes, true);
                self.in_offset += in_bytes;
                self.out_offset += out_bytes;

                match status {
                    InflateStatusNeedsMoreInput | InflateStatusHasMoreOutput => {
                        // The internal out_buf is full; break out to drain output.
                        // Important to process until out_buf is full because the LZ dictionary 
                        // at the beginning of the buffer is being re-used until buf is full.
                        if self.out_offset == out_buf_total {
                            break;
                        }
                        // Otherwise loop back to read more input
                    },
                    InflateStatusDone => {
                        // Break out to drain output.
                        self.decomp_done = true;
                        break;
                    },
                    _ => return Err(status)
                }
            }
        }
    }

    /// Gets the buffer length of the rest_buf, the extra data left over from decompression.
    pub fn get_rest_len(&self) -> uint {
        return self.in_buf_total - self.in_offset;
    }

    /// Gets the rest_buf, the extra data left over from decompression.
    pub fn get_rest(&self, rest_buf: &mut [u8]) -> uint {
        let copy_len = num::min(rest_buf.len(), self.in_buf_total - self.in_offset);
        vec::bytes::copy_memory(rest_buf, self.in_buf.slice(self.in_offset, self.in_buf_total), copy_len);
        return copy_len;
    }

    /// Low level decompress method.  Decompresses DEFLATE-encoded compressed data.
    /// You really need to know what you are doing to call this directly.  It's fragile with edge cases.
    /// It has multiple modes of operation depending on the parameters.
    ///
    /// in_buf has the input data to be decompressed.
    /// in_offset is the offset into in_buf to start reading the data.
    /// in_bytes is the number of bytes to read starting from in_offset, as call input.
    /// in_bytes is the number of bytes has been consumed, as call output.
    /// final_in_data set to true for the last batch of input data, set to false for more calls with more input.
    /// out_buf is the decompressed output data.  The buffer size must be at least MIN_DECOMPRESS_BUF_SIZE and POWER OF 2.
    /// out_offset is the offset into out_buf to start writing the decompressed data.
    /// out_bytes is the number of bytes available to store the decompressed data starting from out_offset, as call input.
    /// out_bytes is the number of bytes has been used up to store the decompressed data, as call output.
    /// reuse_out_buf set to true if reuse out_buf across multiple calls (the decompressed dictionary at the
    /// beginning of the buffer needed to be kept for subsequent calls).  This is typically for using a smaller out_buf
    /// to repeatedly decompress large input data.  Set reuse_out_buf to false if out_buf is not being reused;
    /// typically the buffer is big enough to contain all decompressed data.
    pub fn decompress_buf(&self,
                          in_buf:  &[u8], in_offset:  uint, in_bytes:  &mut uint, final_input: bool, 
                          out_buf: &[u8], out_offset: uint, out_bytes: &mut uint, reuse_out_buf: bool) -> InflateStatus {
        #[fixed_stack_segment];
        #[inline(never)];

        let mut status : c_int = 0;
        let mut in_bytes_sz  = *in_bytes as size_t;
        let mut out_bytes_sz = *out_bytes as size_t;
        let in_buf_next  = in_buf.slice(in_offset, in_offset + *in_bytes);
        let out_buf_next = out_buf.slice(out_offset, out_offset + *out_bytes);
        let decompress_flags: c_uint = 
            if final_input   { 0 } else { TINFL_FLAG_HAS_MORE_INPUT } |
            if reuse_out_buf { 0 } else { TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF };

        do in_buf_next.as_imm_buf |in_next_ptr, _| {
            do out_buf.as_imm_buf |out_base_ptr, _| {
                do out_buf_next.as_imm_buf |out_next_ptr, _| {
                    unsafe {
                        status = rustrt::tinfl_decompress(self.tinfl_decompressor, 
                                                          in_next_ptr as *c_void, 
                                                          &mut in_bytes_sz, 
                                                          out_base_ptr as *c_void, 
                                                          out_next_ptr as *c_void, 
                                                          &mut out_bytes_sz, 
                                                          decompress_flags);
                    }
                }
            }
        }

        *in_bytes = in_bytes_sz as uint;
        *out_bytes = out_bytes_sz as uint;
        return InflateStatus::from_status(status);
    }

}

impl Drop for Inflator {
    fn drop(&mut self) {
        self.free();
    }
}



fn deflate_bytes_internal(bytes: &[u8], flags: c_int) -> ~[u8] {
    #[fixed_stack_segment]; #[inline(never)];

    do bytes.as_imm_buf |b, len| {
        unsafe {
            let mut outsz : size_t = 0;
            let res =
                rustrt::tdefl_compress_mem_to_heap(b as *c_void,
                                                   len as size_t,
                                                   &mut outsz,
                                                   flags);
            assert!(res as int != 0);
            let out = vec::raw::from_buf_raw(res as *u8,
                                             outsz as uint);
            rustrt::mz_free(res);
            out
        }
    }
}

/// Compress a byte buffer to a buffer in heap
pub fn deflate_bytes(bytes: &[u8]) -> ~[u8] {
    deflate_bytes_internal(bytes, LZ_NORM)
}

/// Compress a byte buffer to a buffer in heap with zlib-header
pub fn deflate_bytes_zlib(bytes: &[u8]) -> ~[u8] {
    deflate_bytes_internal(bytes, LZ_NORM | TDEFL_WRITE_ZLIB_HEADER as c_int)
}

fn inflate_bytes_internal(bytes: &[u8], flags: c_int) -> ~[u8] {
    #[fixed_stack_segment]; #[inline(never)];

    do bytes.as_imm_buf |b, len| {
        unsafe {
            let mut outsz : size_t = 0;
            let res =
                rustrt::tinfl_decompress_mem_to_heap(b as *c_void,
                                                     len as size_t,
                                                     &mut outsz,
                                                     flags);
            assert!(res as int != 0);
            let out = vec::raw::from_buf_raw(res as *u8,
                                            outsz as uint);
            rustrt::mz_free(res);
            out
        }
    }
}

/// Decompress a byte buffer to a buffer in heap
pub fn inflate_bytes(bytes: &[u8]) -> ~[u8] {
    inflate_bytes_internal(bytes, 0)
}

/// Decompress a byte buffer with zlib-header to a buffer in heap
pub fn inflate_bytes_zlib(bytes: &[u8]) -> ~[u8] {
    inflate_bytes_internal(bytes, TINFL_FLAG_PARSE_ZLIB_HEADER as c_int)
}



#[cfg(test)]
mod tests {
    use std::rt::io::mem::MemWriter;
    use std::rt::io::mem::MemReader;
    use std::rt::io::Decorator;
    use std::vec;
    use std::num;
    use std::ptr;
    use std::rand;
    use std::rand::Rng;
    use super::Deflator;
    use super::Inflator;
    use super::MIN_DECOMPRESS_BUF_SIZE;
    use super::deflate_bytes;
    use super::inflate_bytes;

    #[test]
    fn test_deflator_alloc() {
        let mut deflator = Deflator::new();
        assert!(( deflator.tdefl_compressor != ptr::null() ));
        deflator.free();
        assert!(( deflator.tdefl_compressor == ptr::null() ));
    }

    #[test]
    fn test_deflator_alloc_multi_free() {
        let mut deflator = Deflator::new();
        assert!(( deflator.tdefl_compressor != ptr::null() ));
        deflator.free();
        assert!(( deflator.tdefl_compressor == ptr::null() ));
        deflator.free();
        assert!(( deflator.tdefl_compressor == ptr::null() ));
    }

    #[test]
    fn test_deflator_init() {
        let deflator = Deflator::new();

        match deflator.init(6, false, false) {
            DeflateStatusOkay => (),
            _ =>  fail!()
        }
    }

    #[test]
    fn test_deflator_reinit() {
        let deflator = Deflator::new();

        match deflator.init(6, false, false) {
            DeflateStatusOkay => (),
            _ =>  fail!()
        }

        match deflator.init(6, false, false) {
            DeflateStatusOkay => (),
            _ =>  fail!()
        }

    }

    #[test]
    fn test_deflator_simple() {
        let mut deflator = Deflator::new();
        deflator.init(6, false, false);

        let in_buf  = bytes!("ABCDEFGHABCDEFGHABCDEFGH");
        let mut in_bytes = in_buf.len();
        let out_buf = vec::from_elem(32, 0u8);
        let mut out_bytes = out_buf.len();
        match deflator.compress_buf(in_buf, 0, &mut in_bytes, out_buf, 0, &mut out_bytes, true) {
            DeflateStatusOkay => (),
            DeflateStatusDone => (),
            _ => fail!()
        }
        deflator.free();

        assert!(( in_bytes == in_buf.len() ));
        assert!(( out_bytes > 0 && out_bytes <= in_bytes ));

    }

    #[test]
    fn test_deflator_multi_input1() {
        let mut deflator = Deflator::new();
        deflator.init(6, false, false);

        // Original in_buf
        let mut in_buf  = bytes!("ABCDEFGHABCDEFGHABCDEFGH");
        let mut in_bytes = in_buf.len();
        let mut out_buf = vec::from_elem(32, 0u8);
        let mut out_bytes = out_buf.len();
        match deflator.compress_buf(in_buf, 0, &mut in_bytes, out_buf, 0, &mut out_bytes, true) {
            DeflateStatusOkay => (),
            DeflateStatusDone => (),
            _ => fail!()
        }
        assert!(( in_bytes == in_buf.len() ));
        assert!(( out_bytes > 0 && out_bytes <= in_bytes ));

        let enc_len = out_bytes;
        let enc_data = out_buf;

        // in_buf part1
        deflator.init(6, false, false);
        in_buf   = bytes!("ABCDEFGH");
        in_bytes = in_buf.len();
        out_buf = vec::from_elem(32, 0u8);
        let mut out_offset = 0;
        out_bytes  = out_buf.len() - out_offset;
        match deflator.compress_buf(in_buf, 0, &mut in_bytes, out_buf, out_offset, &mut out_bytes, false) {
            DeflateStatusOkay => (),
            _ => fail!()
        }
        out_offset += out_bytes;
        out_bytes  = out_buf.len() - out_offset;

        // in_buf part2
        in_buf  = bytes!("ABCDEFGHABCDEFGH");
        in_bytes = in_buf.len();
        match deflator.compress_buf(in_buf, 0, &mut in_bytes, out_buf, out_offset, &mut out_bytes, true) {
            DeflateStatusDone => (),
            _ => fail!()
        }

        let enc_len2 = out_bytes;
        let enc_data2 = out_buf;

        // println(fmt!("enc_data:  %?,  %?", enc_len, enc_data));
        // println(fmt!("enc_data2: %?,  %?", enc_len2, enc_data2));
        assert!(( enc_len == enc_len2 ));
        assert!(( enc_data == enc_data2 ));

        deflator.free();
    }

    #[test]
    fn test_deflator_multi_input2() {
        let mut deflator = Deflator::new();
        deflator.init(6, false, false);

        // Original in_buf
        let in_buf  = bytes!("ABCDEFGHABCDEFGHABCDEFGH");
        let mut in_bytes = in_buf.len();
        let mut out_buf = vec::from_elem(32, 0u8);
        let mut out_bytes = out_buf.len();
        match deflator.compress_buf(in_buf, 0, &mut in_bytes, out_buf, 0, &mut out_bytes, true) {
            DeflateStatusOkay => (),
            DeflateStatusDone => (),
            _ => fail!()
        }
        assert!(( in_bytes == in_buf.len() ));
        assert!(( out_bytes > 0 && out_bytes <= in_bytes ));

        let enc_len = out_bytes;
        let enc_data = out_buf;

        // Same buffer, use in_offset and in_bytes to control the amount of input data to compress.
        deflator.init(6, false, false);
        let in_buf  = bytes!("ABCDEFGHABCDEFGHABCDEFGH");
        let mut in_offset = 0;
        in_bytes = in_buf.len() / 2;
        out_buf = vec::from_elem(32, 0u8);
        let mut out_offset = 0;
        out_bytes  = out_buf.len() - out_offset;
        match deflator.compress_buf(in_buf, in_offset, &mut in_bytes, out_buf, out_offset, &mut out_bytes, false) {
            DeflateStatusOkay => (),
            _ => fail!()
        }
        in_offset += in_bytes;
        in_bytes = in_buf.len() - in_offset;
        out_offset += out_bytes;
        out_bytes  = out_buf.len() - out_offset;

        // Second call with updated in_offset and in_bytes
        match deflator.compress_buf(in_buf, in_offset, &mut in_bytes, out_buf, out_offset, &mut out_bytes, true) {
            DeflateStatusDone => (),
            _ => fail!()
        }

        let enc_len2 = out_bytes;
        let enc_data2 = out_buf;

        // println(fmt!("enc_data:  %?,  %?", enc_len, enc_data));
        // println(fmt!("enc_data2: %?,  %?", enc_len2, enc_data2));
        assert!(( enc_len == enc_len2 ));
        assert!(( enc_data == enc_data2 ));

        deflator.free();
    }

    #[test]
    fn test_deflator_multi_input3() {
        let mut deflator = Deflator::new();
        deflator.init(6, false, false);

        // Original in_buf
        let in_buf  = bytes!("ABCDEFGHABCDEFGHABCDEFGH");
        let mut in_bytes = in_buf.len();
        let mut out_buf = vec::from_elem(32, 0u8);
        let mut out_bytes = out_buf.len();
        match deflator.compress_buf(in_buf, 0, &mut in_bytes, out_buf, 0, &mut out_bytes, true) {
            DeflateStatusOkay => (),
            DeflateStatusDone => (),
            _ => fail!()
        }
        assert!(( in_bytes == in_buf.len() ));
        assert!(( out_bytes > 0 && out_bytes <= in_bytes ));

        let enc_len = out_bytes;
        let enc_data = out_buf;

        // Same buffer, use in_offset and in_bytes to control the amount of input data to compress.
        deflator.init(6, false, false);
        let in_buf  = bytes!("ABCDEFGHABCDEFGHABCDEFGH");
        let mut in_offset = 0;
        in_bytes = in_buf.len() / 2;
        out_buf = vec::from_elem(32, 0u8);
        let mut out_offset = 0;
        out_bytes  = out_buf.len() - out_offset;
        match deflator.compress_buf(in_buf, in_offset, &mut in_bytes, out_buf, out_offset, &mut out_bytes, false) {
            DeflateStatusOkay => (),
            _ => fail!()
        }
        in_offset += in_bytes;
        in_bytes = in_buf.len() - in_offset;
        out_offset += out_bytes;
        out_bytes  = out_buf.len() - out_offset;

        // Second call with updated in_offset and in_bytes
        match deflator.compress_buf(in_buf, in_offset, &mut in_bytes, out_buf, out_offset, &mut out_bytes, false) {
            DeflateStatusOkay => (),
            _ => fail!()
        }
        in_offset += in_bytes;
        in_bytes = in_buf.len() - in_offset;
        out_offset += out_bytes;
        out_bytes  = out_buf.len() - out_offset;

        // Third call with empty input data but with the final_input set to true
        match deflator.compress_buf(in_buf, in_offset, &mut in_bytes, out_buf, out_offset, &mut out_bytes, true) {
            DeflateStatusDone => (),
            _ => fail!()
        }

        let enc_len2 = out_bytes;
        let enc_data2 = out_buf;

        // println(fmt!("enc_data:  %?,  %?", enc_len, enc_data));
        // println(fmt!("enc_data2: %?,  %?", enc_len2, enc_data2));
        assert!(( enc_len == enc_len2 ));
        assert!(( enc_data == enc_data2 ));

        deflator.free();
    }

    #[test]
    fn test_deflator_outbuf_small_outbuf() {
        let mut deflator = Deflator::new();
        deflator.init(6, false, false);

        let in_buf  = bytes!("ABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGH");
        let mut in_bytes = in_buf.len();
        let out_buf = vec::from_elem(4, 0u8);
        let mut out_bytes = out_buf.len();
        // println(fmt!("1. in_bytes: %?", in_bytes));
        let status = deflator.compress_buf(in_buf, 0, &mut in_bytes, out_buf, 0, &mut out_bytes, true);
        match status {
            DeflateStatusOkay => (),
            _ => fail!()
        }
        deflator.free();

        // println(fmt!("1. status: %?", status));
        // println(fmt!("1. in_bytes: %?", in_bytes));
        // println(fmt!("1. out_buf: %?", out_buf));
        // println(fmt!("1. out_bytes: %?", out_bytes));

        // Compression doesn't handle small outbuf very well.  It would just truncate the data not fitted in the outbuf.
        // Use out_bytes equals to the original buffer length as an indicator of running out of room.
        // In general out_buf should be as big as in_buf plus some extra length to ensure capturing all the compressed data.
        assert!(( in_bytes == in_buf.len() ));
        assert!(( out_bytes == out_buf.len() ));

    }

    #[test]
    fn test_deflator_stream() {
        let mut deflator = Deflator::new();
        deflator.init(6, false, false);

        // Compress standard data
        let in_buf  = bytes!("ABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGH").to_owned();
        let mut in_bytes = in_buf.len();
        let out_buf = vec::from_elem(64, 0u8);
        let mut out_bytes = out_buf.len();
        let mut status = deflator.compress_buf(in_buf, 0, &mut in_bytes, out_buf, 0, &mut out_bytes, true);
        match status {
            DeflateStatusDone => (),
            _ => fail!()
        }

        let mut mreader = MemReader::new(in_buf);
        let mut mwriter = MemWriter::new();
        deflator.init(6, false, false);
        status = deflator.compress_stream_rw(&mut mreader, &mut mwriter);
        match status {
            DeflateStatusDone => (),
            _ => fail!()
        }

        let std_out = out_buf.slice(0, out_bytes);
        let cmp_buf = mwriter.inner();
        assert!(( std_out == cmp_buf ));

        deflator.free();
    }


    #[test]
    fn test_inflator_alloc() {
        let mut inflator = Inflator::new();
        assert!(( inflator.tinfl_decompressor != ptr::null() ));
        inflator.free();
        assert!(( inflator.tinfl_decompressor == ptr::null() ));
        inflator.free();
        assert!(( inflator.tinfl_decompressor == ptr::null() ));

        unsafe {
            inflator = Inflator::new();
            let decomp_bytes = vec::raw::from_buf_raw(inflator.tinfl_decompressor as *u8, 32);
            //println(fmt!("Inflator::new(), tinfl_decompressor: %?", decomp_bytes));
            // The first 4 bytes are tinfl_decompressor.m_state, and should be 0
            assert!(( decomp_bytes[0] == 0 && decomp_bytes[1] == 0 && decomp_bytes[2] == 0 && decomp_bytes[3] == 0 ));
        }

    }

    //#[ignore]
    #[test]
    fn test_inflator_extra_byte_bug() {
        let mut comp = Deflator::new();
        comp.init(10, false, false);

        let in_buf  = bytes!("ABCDEFGH\r\n");
        let mut in_bytes = in_buf.len();
        let comp_buf = vec::from_elem(64, 0u8);
        let mut comp_bytes = comp_buf.len();
        match comp.compress_buf(in_buf, 0, &mut in_bytes, comp_buf, 0, &mut comp_bytes, true) {
            DeflateStatusOkay => (),
            DeflateStatusDone => (),
            _ => fail!()
        }
        comp.free();

        let comp_buf = ~[0x73, 0x74, 0x72, 0x76, 0x71, 0x75, 0x73, 0xF7, 0xE0, 0xE5, 0x02, 0x00, 0x94, 0xA6, 0xD7, 0xD0, 0x0A, 0x00, 0x00, 0x00];
        //println(format!("1: comp_buf: {:?}", comp_buf));

        let mut inflator = Inflator::new();
        let de_in_total = comp_buf.len();
        let mut de_in_bytes = de_in_total;
        let decomp_buf = vec::from_elem(MIN_DECOMPRESS_BUF_SIZE, 0u8);
        let mut decomp_bytes = decomp_buf.len();
        match inflator.decompress_buf(comp_buf, 0, &mut de_in_bytes, true, decomp_buf, 0, &mut decomp_bytes, false) {
            InflateStatusDone => (),
            _ => fail!()
        }
        inflator.free();

        // let decomp_data = decomp_buf.slice(0, decomp_bytes);
        // println(format!("1: decomp_data: {:?}", decomp_data));
        // println(format!("1: de_in_bytes: {:?}", de_in_bytes));
        // println(format!("1: de_in_total: {:?}", de_in_total));

    }

    #[test]
    fn test_inflator_simple() {
        let mut comp = Deflator::new();
        comp.init(6, false, false);

        let in_buf  = bytes!("ABCDEFGH");
        let mut in_bytes = in_buf.len();
        let comp_buf = vec::from_elem(64, 0u8);
        let mut comp_bytes = comp_buf.len();
        match comp.compress_buf(in_buf, 0, &mut in_bytes, comp_buf, 0, &mut comp_bytes, true) {
            DeflateStatusOkay => (),
            DeflateStatusDone => (),
            _ => fail!()
        }
        comp.free();

        let mut inflator = Inflator::new();
        let mut de_in_bytes = comp_bytes;
        let decomp_buf = vec::from_elem(MIN_DECOMPRESS_BUF_SIZE, 0u8);
        let mut decomp_bytes = decomp_buf.len();
        match inflator.decompress_buf(comp_buf, 0, &mut de_in_bytes, true, decomp_buf, 0, &mut decomp_bytes, false) {
            InflateStatusDone => (),
            _ => fail!()
        }
        inflator.free();

        let decomp_data = decomp_buf.slice(0, decomp_bytes);
        assert!(( in_buf == decomp_data ));

    }

    #[test]
    fn test_inflator_big_data_one_pass() {
        let mut comp = Deflator::new();
        comp.init(6, false, false);

        let mut rnd = rand::rng();
        let mut words = ~[];
        do 2000.times {
            let range = rnd.gen_range(1u, 10);
            words.push(rnd.gen_vec::<u8>(range));
        }

        let in_buf  = words.concat_vec();
        let mut in_bytes = in_buf.len();
        let comp_buf = vec::from_elem(in_bytes * 2, 0u8);
        let mut comp_bytes = comp_buf.len();
        let status = comp.compress_buf(in_buf, 0, &mut in_bytes, comp_buf, 0, &mut comp_bytes, true);
        match status {
            DeflateStatusDone => (),
            _ => fail!()
        }
        comp.free();

        //println(format!("in_buf: {:?}", in_buf.len()));
        //println(format!("2. status: {:?}", status));
        //println(format!("2. in_bytes: {:?}", in_bytes));
        //println(format!("2. comp_bytes: {:?}", comp_bytes));

        let mut inflator = Inflator::new();
        let mut de_in_bytes = comp_bytes;
        let decomp_buf = vec::from_elem(in_bytes, 0u8);
        let mut decomp_bytes = decomp_buf.len();
        let status = inflator.decompress_buf(comp_buf, 0, &mut de_in_bytes, true, decomp_buf, 0, &mut decomp_bytes, false);
        match status {
            InflateStatusDone => (),
            InflateStatusHasMoreOutput => { println("Has more output."); fail!(); },
            _ => fail!()
        }
        inflator.free();

        let decomp_data = decomp_buf.slice(0, decomp_bytes).to_owned();
        assert!(( in_buf == decomp_data ));

    }

    #[test]
    fn test_inflator_single_inbuf_multi_outbuf() {
        let mut comp = Deflator::new();
        comp.init(6, false, false);

        let mut rnd = rand::rng();
        let mut words = ~[];
        do 20000.times {
            let range = rnd.gen_range(1u, 10);
            words.push(rnd.gen_vec::<u8>(range));
        }

        let in_buf  = words.concat_vec();
        let mut in_bytes = in_buf.len();
        let comp_buf = vec::from_elem(in_bytes * 2, 0u8);
        let mut comp_bytes = comp_buf.len();
        let status = comp.compress_buf(in_buf, 0, &mut in_bytes, comp_buf, 0, &mut comp_bytes, true);
        match status {
            DeflateStatusOkay => (),
            DeflateStatusDone => (),
            _ => fail!()
        }
        comp.free();

        // println(format!("2. in_buf: {:?}", in_buf.len()));
        // println(format!("2. status: {:?}", status));
        // println(format!("2. in_bytes: {:?}", in_bytes));
        // println(format!("2. comp_bytes: {:?}", comp_bytes));

        let mut inflator = Inflator::new();
        let de_in_total = comp_bytes;
        let mut de_in_offset = 0;
        let mut de_in_bytes;
        let mut decomp_data : ~[u8] = ~[];
        let decomp_buf = vec::from_elem(MIN_DECOMPRESS_BUF_SIZE, 0u8);
        let decomp_total = decomp_buf.len();
        let mut decomp_offset = 0;
        let mut decomp_bytes;
        loop {
            de_in_bytes = de_in_total - de_in_offset;
            decomp_bytes = decomp_total - decomp_offset;
            let status = inflator.decompress_buf(comp_buf, de_in_offset, &mut de_in_bytes, true, decomp_buf, decomp_offset, &mut decomp_bytes, true);
            // println(format!("de: status: {:?}", status));
            // println(format!("de: de_in_offset: {:?}", de_in_offset));
            // println(format!("de: de_in_bytes: {:?}", de_in_bytes));
            // println(format!("de: de_in_total: {:?}", de_in_total));
            // println(format!("de: decomp_offset: {:?}", decomp_offset));
            // println(format!("de: decomp_bytes: {:?}", decomp_bytes));
            // println(format!("de: decomp_total: {:?}", decomp_total));

            de_in_offset += de_in_bytes;
            decomp_offset += decomp_bytes;

            match status {
                InflateStatusDone => {
                    decomp_data.push_all(decomp_buf.slice(0, decomp_offset));
                    break;
                },
                InflateStatusHasMoreOutput => { 
                    //println("de: Has more output...");
                    if decomp_offset == decomp_total {
                        // output decomp_buf is full.  flush its content to the accumulator buffer.  Reset decomp_buf.
                        decomp_data.push_all(decomp_buf);
                        decomp_offset = 0;
                    }
                },
                InflateStatusNeedsMoreInput => {
                    fail!(format!("Decompression unexpected status.  status: {:?}", status))
                },
                _ => fail!(format!("Decompression failed.  status: {:?}", status))
            }
        }

        assert!(( in_buf == decomp_data ));

        inflator.free();
    }

    #[test]
    fn test_inflator_multi_inbuf_multi_outbuf() {
        let mut comp = Deflator::new();
        comp.init(6, false, false);

        let mut rnd = rand::rng();
        let mut words = ~[];
        do 20000.times {
            let range = rnd.gen_range(1u, 10);
            words.push(rnd.gen_vec::<u8>(range));
        }

        let in_buf  = words.concat_vec();
        let mut in_bytes = in_buf.len();
        let comp_buf = vec::from_elem(in_bytes * 2, 0u8);
        let mut comp_bytes = comp_buf.len();
        let status = comp.compress_buf(in_buf, 0, &mut in_bytes, comp_buf, 0, &mut comp_bytes, true);
        match status {
            DeflateStatusOkay => (),
            DeflateStatusDone => (),
            _ => fail!()
        }
        comp.free();

        // println(format!("2. in_buf: {:?}", in_buf.len()));
        // println(format!("2. status: {:?}", status));
        // println(format!("2. in_bytes: {:?}", in_bytes));
        // println(format!("2. comp_bytes: {:?}", comp_bytes));

        let mut inflator = Inflator::new();
        let de_in_total = comp_bytes;
        let de_in_batch_size = 16*1024u;
        let mut de_in_offset = 0;
        let mut de_in_bytes;
        let mut decomp_data : ~[u8] = ~[];
        let decomp_buf = vec::from_elem(MIN_DECOMPRESS_BUF_SIZE, 0u8);
        let decomp_total = decomp_buf.len();
        let mut decomp_offset = 0;
        let mut decomp_bytes;
        loop {
            de_in_bytes = num::min(de_in_total - de_in_offset, de_in_batch_size);   // limit in_bytes to a smaller batch to simulate multiple in_buf
            decomp_bytes = decomp_total - decomp_offset;
            let final_input = de_in_offset + de_in_bytes == de_in_total;
            let status = inflator.decompress_buf(comp_buf, de_in_offset, &mut de_in_bytes, final_input, decomp_buf, decomp_offset, &mut decomp_bytes, true);
            // println(format!("de: status: {:?}", status));
            // println(format!("de: de_in_offset: {:?}", de_in_offset));
            // println(format!("de: de_in_bytes: {:?}", de_in_bytes));
            // println(format!("de: de_in_total: {:?}", de_in_total));
            // println(format!("de: decomp_offset: {:?}", decomp_offset));
            // println(format!("de: decomp_bytes: {:?}", decomp_bytes));
            // println(format!("de: decomp_total: {:?}", decomp_total));

            de_in_offset += de_in_bytes;
            decomp_offset += decomp_bytes;

            match status {
                InflateStatusDone => {
                    decomp_data.push_all(decomp_buf.slice(0, decomp_offset));
                    break;
                },
                InflateStatusHasMoreOutput => { 
                    //println("de: Has more output...");
                    if decomp_offset == decomp_total {
                        // output decomp_buf is full.  flush its content to the accumulator buffer.  Reset decomp_buf.
                        decomp_data.push_all(decomp_buf);
                        decomp_offset = 0;
                    }
                },
                InflateStatusNeedsMoreInput => {
                    //println("de: Need more input......");
                },
                _ => fail!(format!("Decompression failed.  status: {:?}", status))
            }
        }

        assert!(( in_buf == decomp_data ));

        inflator.free();
    }

    #[test]
    fn test_inflator_stream() {
        let mut comp = Deflator::new();
        comp.init(6, false, false);

        // Compress standard data
        let in_buf  = bytes!("ABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGH").to_owned();
        let mut in_bytes = in_buf.len();
        let out_buf = vec::from_elem(64, 0u8);
        let mut out_bytes = out_buf.len();
        let status = comp.compress_buf(in_buf, 0, &mut in_bytes, out_buf, 0, &mut out_bytes, true);
        match status {
            DeflateStatusDone => (),
            _ => fail!()
        }
        let comp_buf = out_buf.slice(0, out_bytes);
        comp.free();

        let mut mreader = MemReader::new(comp_buf.to_owned());
        let mut mwriter = MemWriter::new();
        let mut inflator = Inflator::new();
        let status = inflator.decompress_stream_rw(&mut mreader, &mut mwriter);
        match status {
            InflateStatusDone => (),
            _ => fail!()
        }

        let cmp_buf = mwriter.inner();
        assert!(( in_buf == cmp_buf ));

        inflator.free();
    }

    #[test]
    fn test_inflator_corrupted_data() {
        let mut comp = Deflator::new();
        comp.init(6, false, false);

        let in_buf  = bytes!("ABCDEFGH");
        let mut in_bytes = in_buf.len();
        let comp_buf = vec::from_elem(64, 0u8);
        let mut comp_bytes = comp_buf.len();
        match comp.compress_buf(in_buf, 0, &mut in_bytes, comp_buf, 0, &mut comp_bytes, true) {
            DeflateStatusOkay => (),
            DeflateStatusDone => (),
            _ => fail!()
        }
        comp.free();

        let mut inflator = Inflator::new();
        let mut de_in_bytes = comp_bytes - 1;    // missing one byte;
        let decomp_buf = vec::from_elem(MIN_DECOMPRESS_BUF_SIZE, 0u8);
        let mut decomp_bytes = decomp_buf.len();
        let status = inflator.decompress_buf(comp_buf, 0, &mut de_in_bytes, true, decomp_buf, 0, &mut decomp_bytes, false);
        //println(format!("status: {:?}", status));
        match status {
            InflateStatusDone =>  fail!("Corrupted data should not work"),
            InflateStatusFailed | _  => (),
        }
        inflator.free();

    }


    #[test]
    fn test_inflator_decompress_read_out_len_1() {
        let mut comp = Deflator::new();
        comp.init(6, false, false);

        // Compress standard data
        let in_buf  = bytes!("ABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGH").to_owned();
        let mut in_bytes = in_buf.len();
        let out_buf = vec::from_elem(64, 0u8);
        let mut out_bytes = out_buf.len();
        let status = comp.compress_buf(in_buf, 0, &mut in_bytes, out_buf, 0, &mut out_bytes, true);
        match status {
            DeflateStatusDone => (),
            _ => fail!()
        }
        let comp_buf = out_buf.slice(0, out_bytes);
        comp.free();

        let mut mreader = MemReader::new(comp_buf.to_owned());
        let mut inflator = Inflator::new();
        let mut output_buf = vec::from_elem(1, 0u8);
        let mut decomp_buf : ~[u8] = ~[];
        loop {
            let retval = inflator.decompress_read(
                |in_buf| {
                    if mreader.eof() {
                        0                           // Return 0 for EOF
                    } else {
                        match mreader.read(in_buf) {
                            Some(nread) => nread,   // Return number of bytes read, including 0 for EOF
                            None => 0               // REturn 0 for EOF
                        }
                    }
                },
                output_buf);
            //println(format!("retval: {:?}", retval));
            match retval {
                Ok(0) => 
                    break,
                Ok(output_len) => 
                    decomp_buf.push_all(output_buf.slice(0, output_len)),
                _ => 
                    fail!(format!("retval: {:?}", retval))
            }
        }

        assert!(( in_buf == decomp_buf ));

        inflator.free();
    }

    #[test]
    fn test_inflator_decompress_read_out_len_8() {
        let mut comp = Deflator::new();
        comp.init(6, false, false);

        // Compress standard data
        let in_buf  = bytes!("ABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGH").to_owned();
        let mut in_bytes = in_buf.len();
        let out_buf = vec::from_elem(64, 0u8);
        let mut out_bytes = out_buf.len();
        let status = comp.compress_buf(in_buf, 0, &mut in_bytes, out_buf, 0, &mut out_bytes, true);
        match status {
            DeflateStatusDone => (),
            _ => fail!()
        }
        let comp_buf = out_buf.slice(0, out_bytes);
        comp.free();

        let mut mreader = MemReader::new(comp_buf.to_owned());
        let mut inflator = Inflator::new();
        let mut output_buf = vec::from_elem(8, 0u8);
        let mut decomp_buf : ~[u8] = ~[];
        loop {
            let retval = inflator.decompress_read(
                |in_buf| {
                    if mreader.eof() {
                        0                           // Return 0 for EOF
                    } else {
                        match mreader.read(in_buf) {
                            Some(nread) => nread,   // Return number of bytes read, including 0 for EOF
                            None => 0               // REturn 0 for EOF
                        }
                    }
                },
                output_buf);
            //println(format!("retval: {:?}", retval));
            match retval {
                Ok(0) => 
                    break,
                Ok(output_len) => 
                    decomp_buf.push_all(output_buf.slice(0, output_len)),
                _ => 
                    fail!(format!("retval: {:?}", retval))
            }
        }

        assert!(( in_buf == decomp_buf ));

        inflator.free();
    }

    #[test]
    fn test_inflator_decompress_read_out_len_all() {
        let mut comp = Deflator::new();
        comp.init(6, false, false);

        // Compress standard data
        let in_buf  = bytes!("ABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGH").to_owned();
        let mut in_bytes = in_buf.len();
        let out_buf = vec::from_elem(64, 0u8);
        let mut out_bytes = out_buf.len();
        let status = comp.compress_buf(in_buf, 0, &mut in_bytes, out_buf, 0, &mut out_bytes, true);
        match status {
            DeflateStatusDone => (),
            _ => fail!()
        }
        let comp_buf = out_buf.slice(0, out_bytes);
        comp.free();

        let mut mreader = MemReader::new(comp_buf.to_owned());
        let mut inflator = Inflator::new();
        let mut output_buf = vec::from_elem(256, 0u8);
        let mut decomp_buf : ~[u8] = ~[];
        loop {
            let retval = inflator.decompress_read(
                |in_buf| {
                    if mreader.eof() {
                        0                           // Return 0 for EOF
                    } else {
                        match mreader.read(in_buf) {
                            Some(nread) => nread,   // Return number of bytes read, including 0 for EOF
                            None => 0               // REturn 0 for EOF
                        }
                    }
                },
                output_buf);
            //println(format!("retval: {:?}", retval));
            match retval {
                Ok(0) => 
                    break,
                Ok(output_len) => 
                    decomp_buf.push_all(output_buf.slice(0, output_len)),
                _ => 
                    fail!(format!("retval: {:?}", retval))
            }
        }

        assert!(( in_buf == decomp_buf ));

        inflator.free();
    }




    #[test]
    fn test_flate_round_trip() {
        let mut r = rand::rng();
        let mut words = ~[];
        do 20.times {
            let range = r.gen_range(1u, 10);
            words.push(r.gen_vec::<u8>(range));
        }
        do 20.times {
            let mut input = ~[];
            do 2000.times {
                input.push_all(r.choose(words));
            }
            debug!("de/inflate of {} bytes of random word-sequences",
                   input.len());
            let cmp = deflate_bytes(input);
            let out = inflate_bytes(cmp);
            debug!("{} bytes deflated to {} ({:.1f}% size)",
                   input.len(), cmp.len(),
                   100.0 * ((cmp.len() as f64) / (input.len() as f64)));
            assert_eq!(input, out);
        }
    }

    #[test]
    fn test_zlib_flate() {
        let bytes = ~[1, 2, 3, 4, 5];
        let deflated = deflate_bytes(bytes);
        let inflated = inflate_bytes(deflated);
        assert_eq!(inflated, bytes);
    }

}

