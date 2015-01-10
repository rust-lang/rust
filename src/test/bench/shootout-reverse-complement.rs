// The Computer Language Benchmarks Game
// http://benchmarksgame.alioth.debian.org/
//
// contributed by the Rust Project Developers

// Copyright (c) 2013-2014 The Rust Project Developers
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in
//   the documentation and/or other materials provided with the
//   distribution.
//
// - Neither the name of "The Computer Language Benchmarks Game" nor
//   the name of "The Computer Language Shootout Benchmarks" nor the
//   names of its contributors may be used to endorse or promote
//   products derived from this software without specific prior
//   written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
// OF THE POSSIBILITY OF SUCH DAMAGE.

// ignore-android see #10393 #13206

#![feature(unboxed_closures)]

extern crate libc;

use std::io::stdio::{stdin_raw, stdout_raw};
use std::io::{IoResult, EndOfFile};
use std::ptr::{copy_memory, Unique};
use std::thread::Thread;

struct Tables {
    table8: [u8;1 << 8],
    table16: [u16;1 << 16]
}

impl Tables {
    fn new() -> Tables {
        let mut table8 = [0;1 << 8];
        for (i, v) in table8.iter_mut().enumerate() {
            *v = Tables::computed_cpl8(i as u8);
        }
        let mut table16 = [0;1 << 16];
        for (i, v) in table16.iter_mut().enumerate() {
            *v = (table8[i & 255] as u16) << 8 |
                 table8[i >> 8]  as u16;
        }
        Tables { table8: table8, table16: table16 }
    }

    fn computed_cpl8(c: u8) -> u8 {
        match c {
            b'A' | b'a' => b'T',
            b'C' | b'c' => b'G',
            b'G' | b'g' => b'C',
            b'T' | b't' => b'A',
            b'U' | b'u' => b'A',
            b'M' | b'm' => b'K',
            b'R' | b'r' => b'Y',
            b'W' | b'w' => b'W',
            b'S' | b's' => b'S',
            b'Y' | b'y' => b'R',
            b'K' | b'k' => b'M',
            b'V' | b'v' => b'B',
            b'H' | b'h' => b'D',
            b'D' | b'd' => b'H',
            b'B' | b'b' => b'V',
            b'N' | b'n' => b'N',
            i => i,
        }
    }

    /// Retreives the complement for `i`.
    fn cpl8(&self, i: u8) -> u8 {
        self.table8[i as uint]
    }

    /// Retreives the complement for `i`.
    fn cpl16(&self, i: u16) -> u16 {
        self.table16[i as uint]
    }
}

/// Reads all remaining bytes from the stream.
fn read_to_end<R: Reader>(r: &mut R) -> IoResult<Vec<u8>> {
    // As reading the input stream in memory is a bottleneck, we tune
    // Reader::read_to_end() with a fast growing policy to limit
    // recopies.  If MREMAP_RETAIN is implemented in the linux kernel
    // and jemalloc use it, this trick will become useless.
    const CHUNK: uint = 64 * 1024;

    let mut vec = Vec::with_capacity(CHUNK);
    loop {
        // workaround: very fast growing
        let len = vec.len();
        if vec.capacity() - len < CHUNK {
            let cap = vec.capacity();
            let mult = if cap < 256 * 1024 * 1024 {
                16
            } else {
                2
            };
            vec.reserve_exact(mult * cap - len);
        }
        match r.push_at_least(1, CHUNK, &mut vec) {
            Ok(_) => {}
            Err(ref e) if e.kind == EndOfFile => break,
            Err(e) => return Err(e)
        }
    }
    Ok(vec)
}

/// Finds the first position at which `b` occurs in `s`.
fn memchr(h: &[u8], n: u8) -> Option<uint> {
    use libc::{c_void, c_int, size_t};
    let res = unsafe {
        libc::memchr(h.as_ptr() as *const c_void, n as c_int, h.len() as size_t)
    };
    if res.is_null() {
        None
    } else {
        Some(res as uint - h.as_ptr() as uint)
    }
}

/// A mutable iterator over DNA sequences
struct MutDnaSeqs<'a> { s: &'a mut [u8] }
fn mut_dna_seqs<'a>(s: &'a mut [u8]) -> MutDnaSeqs<'a> {
    MutDnaSeqs { s: s }
}
impl<'a> Iterator for MutDnaSeqs<'a> {
    type Item = &'a mut [u8];

    fn next(&mut self) -> Option<&'a mut [u8]> {
        let tmp = std::mem::replace(&mut self.s, &mut []);
        let tmp = match memchr(tmp, b'\n') {
            Some(i) => tmp.slice_from_mut(i + 1),
            None => return None,
        };
        let (seq, tmp) = match memchr(tmp, b'>') {
            Some(i) => tmp.split_at_mut(i),
            None => {
                let len = tmp.len();
                tmp.split_at_mut(len)
            }
        };
        self.s = tmp;
        Some(seq)
    }
}

/// Length of a normal line without the terminating \n.
const LINE_LEN: uint = 60;

/// Compute the reverse complement.
fn reverse_complement(seq: &mut [u8], tables: &Tables) {
    let seq = seq.init_mut();// Drop the last newline
    let len = seq.len();
    let off = LINE_LEN - len % (LINE_LEN + 1);
    let mut i = LINE_LEN;
    while i < len {
        unsafe {
            copy_memory(seq.as_mut_ptr().offset((i - off + 1) as int),
                        seq.as_ptr().offset((i - off) as int), off);
            *seq.get_unchecked_mut(i - off) = b'\n';
        }
        i += LINE_LEN + 1;
    }

    let div = len / 4;
    let rem = len % 4;
    unsafe {
        let mut left = seq.as_mut_ptr() as *mut u16;
        // This is slow if len % 2 != 0 but still faster than bytewise operations.
        let mut right = seq.as_mut_ptr().offset(len as int - 2) as *mut u16;
        let end = left.offset(div as int);
        while left != end {
            let tmp = tables.cpl16(*left);
            *left = tables.cpl16(*right);
            *right = tmp;
            left = left.offset(1);
            right = right.offset(-1);
        }

        let end = end as *mut u8;
        match rem {
            1 => *end = tables.cpl8(*end),
            2 => {
                let tmp = tables.cpl8(*end);
                *end = tables.cpl8(*end.offset(1));
                *end.offset(1) = tmp;
            },
            3 => {
                *end.offset(1) = tables.cpl8(*end.offset(1));
                let tmp = tables.cpl8(*end);
                *end = tables.cpl8(*end.offset(2));
                *end.offset(2) = tmp;
            },
            _ => { },
        }
    }
}


struct Racy<T>(T);

unsafe impl<T: 'static> Send for Racy<T> {}

/// Executes a closure in parallel over the given iterator over mutable slice.
/// The closure `f` is run in parallel with an element of `iter`.
fn parallel<'a, I, T, F>(iter: I, f: F)
        where T: 'a+Send + Sync,
              I: Iterator<Item=&'a mut [T]>,
              F: Fn(&mut [T]) + Sync {
    use std::mem;
    use std::raw::Repr;

    iter.map(|chunk| {
        // Need to convert `f` and `chunk` to something that can cross the task
        // boundary.
        let f = Racy(&f as *const F as *const uint);
        let raw = Racy(chunk.repr());
        Thread::scoped(move|| {
            let f = f.0 as *const F;
            unsafe { (*f)(mem::transmute(raw.0)) }
        })
    }).collect::<Vec<_>>();
}

fn main() {
    let mut data = read_to_end(&mut stdin_raw()).unwrap();
    let tables = &Tables::new();
    parallel(mut_dna_seqs(data.as_mut_slice()), |&: seq| reverse_complement(seq, tables));
    stdout_raw().write(data.as_mut_slice()).unwrap();
}
