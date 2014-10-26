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

// ignore-android doesn't terminate?

#![feature(slicing_syntax, asm, if_let, tuple_indexing)]

extern crate libc;

use std::io::stdio::{stdin_raw, stdout_raw};
use std::sync::{Future};
use std::num::{div_rem};
use std::ptr::{copy_memory};
use std::io::{IoResult, EndOfFile};
use std::slice::raw::{mut_buf_as_slice};

use shared_memory::{SharedMemory};

mod tables {
    use std::sync::{Once, ONCE_INIT};

    /// Lookup tables.
    static mut CPL16: [u16, ..1 << 16] = [0, ..1 << 16];
    static mut CPL8:  [u8,  ..1 << 8]  = [0, ..1 << 8];

    /// Generates the tables.
    pub fn get() -> Tables {
        /// To make sure we initialize the tables only once.
        static INIT: Once = ONCE_INIT;
        INIT.doit(|| {
            unsafe {
                for i in range(0, 1 << 8) {
                    CPL8[i] = match i as u8 {
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
                    };
                }

                for (i, v) in CPL16.iter_mut().enumerate() {
                    *v = *CPL8.unsafe_get(i & 255) as u16 << 8 |
                         *CPL8.unsafe_get(i >> 8)  as u16;
                }
            }
        });
        Tables { _dummy: () }
    }

    /// Accessor for the static arrays.
    ///
    /// To make sure that the tables can't be accessed without having been initialized.
    pub struct Tables {
        _dummy: ()
    }

    impl Tables {
        /// Retreives the complement for `i`.
        pub fn cpl8(self, i: u8) -> u8 {
            // Not really unsafe.
            unsafe { CPL8[i as uint] }
        }

        /// Retreives the complement for `i`.
        pub fn cpl16(self, i: u16) -> u16 {
            unsafe { CPL16[i as uint] }
        }
    }
}

mod shared_memory {
    use std::sync::{Arc};
    use std::mem::{transmute};
    use std::raw::{Slice};

    /// Structure for sharing disjoint parts of a vector mutably across tasks.
    pub struct SharedMemory {
        ptr: Arc<Vec<u8>>,
        start: uint,
        len: uint,
    }

    impl SharedMemory {
        pub fn new(ptr: Vec<u8>) -> SharedMemory {
            let len = ptr.len();
            SharedMemory {
                ptr: Arc::new(ptr),
                start: 0,
                len: len,
            }
        }

        pub fn as_mut_slice(&mut self) -> &mut [u8] {
            unsafe {
                transmute(Slice {
                    data: self.ptr.as_ptr().offset(self.start as int) as *const u8,
                    len: self.len,
                })
            }
        }

        pub fn len(&self) -> uint {
            self.len
        }

        pub fn split_at(self, mid: uint) -> (SharedMemory, SharedMemory) {
            assert!(mid <= self.len);
            let left = SharedMemory {
                ptr: self.ptr.clone(),
                start: self.start,
                len: mid,
            };
            let right = SharedMemory {
                ptr: self.ptr,
                start: self.start + mid,
                len: self.len - mid,
            };
            (left, right)
        }

        /// Resets the object so that it covers the whole range of the contained vector.
        ///
        /// You must not call this method if `self` is not the only reference to the
        /// shared memory.
        ///
        /// FIXME: If `Arc` had a method to check if the reference is unique, then we
        /// wouldn't need the `unsafe` here.
        ///
        /// FIXME: If `Arc` had a method to unwrap the contained value, then we could
        /// simply unwrap here.
        pub unsafe fn reset(self) -> SharedMemory {
            let len = self.ptr.len();
            SharedMemory {
                ptr: self.ptr,
                start: 0,
                len: len,
            }
        }
    }
}


/// Reads all remaining bytes from the stream.
fn read_to_end<R: Reader>(r: &mut R) -> IoResult<Vec<u8>> {
    const CHUNK: uint = 64 * 1024;

    let mut vec = Vec::with_capacity(1024 * 1024);
    loop {
        if vec.capacity() - vec.len() < CHUNK {
            let cap = vec.capacity();
            let mult = if cap < 256 * 1024 * 1024 {
                // FIXME (mahkoh): Temporary workaround for jemalloc on linux. Replace
                // this by 2x once the jemalloc preformance issue has been fixed.
                16
            } else {
                2
            };
            vec.reserve_exact(mult * cap);
        }
        unsafe {
            let ptr = vec.as_mut_ptr().offset(vec.len() as int);
            match mut_buf_as_slice(ptr, CHUNK, |s| r.read(s)) {
                Ok(n) => {
                    let len = vec.len();
                    vec.set_len(len + n);
                },
                Err(ref e) if e.kind == EndOfFile => break,
                Err(e) => return Err(e),
            }
        }
    }
    Ok(vec)
}

/// Finds the first position at which `b` occurs in `s`.
fn memchr(h: &[u8], n: u8) -> Option<uint> {
    use libc::{c_void, c_int, size_t};
    extern {
        fn memchr(h: *const c_void, n: c_int, s: size_t) -> *mut c_void;
    }
    let res = unsafe {
        memchr(h.as_ptr() as *const c_void, n as c_int, h.len() as size_t)
    };
    if res.is_null() {
        None
    } else {
        Some(res as uint - h.as_ptr() as uint)
    }
}

/// Length of a normal line without the terminating \n.
const LINE_LEN: uint = 60;

/// Compute the reverse complement.
fn reverse_complement(mut view: SharedMemory, tables: tables::Tables) {
    // Drop the last newline
    let seq = view.as_mut_slice().init_mut();
    let len = seq.len();
    let off = LINE_LEN - len % (LINE_LEN + 1);
    let mut i = LINE_LEN;
    while i < len {
        unsafe {
            copy_memory(seq.as_mut_ptr().offset((i - off + 1) as int),
                        seq.as_ptr().offset((i - off) as int), off);
            *seq.unsafe_mut(i - off) = b'\n';
        }
        i += LINE_LEN + 1;
    }

    let (div, rem) = div_rem(len, 4);
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

fn main() {
    let mut data = SharedMemory::new(read_to_end(&mut stdin_raw()).unwrap());
    let tables = tables::get();

    let mut futures = vec!();
    loop {
        let (_, mut tmp_data) = match memchr(data.as_mut_slice(), b'\n') {
            Some(i) => data.split_at(i + 1),
            _ => break,
        };
        let (view, tmp_data) = match memchr(tmp_data.as_mut_slice(), b'>') {
            Some(i) => tmp_data.split_at(i),
            None => {
                let len = tmp_data.len();
                tmp_data.split_at(len)
            },
        };
        futures.push(Future::spawn(proc() reverse_complement(view, tables)));
        data = tmp_data;
    }

    for f in futures.iter_mut() {
        f.get();
    }

    // Not actually unsafe. If Arc had a way to check uniqueness then we could do that in
    // `reset` and it would tell us that, yes, it is unique at this point.
    data = unsafe { data.reset() };

    stdout_raw().write(data.as_mut_slice()).unwrap();
}
