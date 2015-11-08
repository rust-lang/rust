// The Computer Language Benchmarks Game
// http://benchmarksgame.alioth.debian.org/
//
// contributed by the Rust Project Developers

// Copyright (c) 2012-2014 The Rust Project Developers
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

use std::cmp::min;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::thread;


const LINE_LEN: usize = 60;

const BLOCK_LINES: usize = 512;
const BLOCK_THOROUGHPUT: usize = LINE_LEN * BLOCK_LINES;
const BLOCK_LEN: usize = BLOCK_THOROUGHPUT + BLOCK_LINES;

const STDIN_BUF: usize = (LINE_LEN + 1) * 1024;


const ALU: &'static [u8] =
    b"GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGG\
      GAGGCCGAGGCGGGCGGATCACCTGAGGTCAGGAGTTCGAGA\
      CCAGCCTGGCCAACATGGTGAAACCCCGTCTCTACTAAAAAT\
      ACAAAAATTAGCCGGGCGTGGTGGCGCGCGCCTGTAATCCCA\
      GCTACTCGGGAGGCTGAGGCAGGAGAATCGCTTGAACCCGGG\
      AGGCGGAGGTTGCAGTGAGCCGAGATCGCGCCACTGCACTCC\
      AGCCTGGGCGACAGAGCGAGACTCCGTCTCAAAAA";

const IUB: &'static [(u8, f32)] =
    &[(b'a', 0.27), (b'c', 0.12), (b'g', 0.12),
      (b't', 0.27), (b'B', 0.02), (b'D', 0.02),
      (b'H', 0.02), (b'K', 0.02), (b'M', 0.02),
      (b'N', 0.02), (b'R', 0.02), (b'S', 0.02),
      (b'V', 0.02), (b'W', 0.02), (b'Y', 0.02)];

const HOMOSAPIENS: &'static [(u8, f32)] =
    &[(b'a', 0.3029549426680),
      (b'c', 0.1979883004921),
      (b'g', 0.1975473066391),
      (b't', 0.3015094502008)];


// We need a specific Rng,
// so implement this manually
const MODULUS: u32 = 139968;
const MULTIPLIER: u32 = 3877;
const ADDITIVE: u32 = 29573;

// Why doesn't rust already have this?
// Algorithm directly taken from Wikipedia
fn powmod(mut base: u64, mut exponent: u32, modulus: u64) -> u64 {
    let mut ret = 1;
    base %= modulus;

    while exponent > 0 {
        if exponent & 1 == 1 {
           ret *= base;
           ret %= modulus;
        }
        exponent >>= 1;
        base *= base;
        base %= modulus;
    }

    ret
}

// Just a typical LCRNG
pub struct Rng {
    last: u32
}

impl Rng {
    pub fn new() -> Rng {
        Rng { last: 42 }
    }

    pub fn max_value() -> u32 {
        MODULUS - 1
    }

    pub fn normalize(p: f32) -> u32 {
        (p * MODULUS as f32).floor() as u32
    }

    pub fn gen(&mut self) -> u32 {
        self.last = (self.last * MULTIPLIER + ADDITIVE) % MODULUS;
        self.last
    }

    // This allows us to fast-forward the RNG,
    // allowing us to run it in parallel.
    pub fn future(&self, n: u32) -> Rng {
        let a = MULTIPLIER as u64;
        let b = ADDITIVE as u64;
        let m = MODULUS as u64;

        //                          (a^n - 1) mod (a-1) m
        // x_k = ((a^n x_0 mod m) + --------------------- b) mod m
        //                                   a - 1
        //
        // Since (a - 1) divides (a^n - 1) mod (a-1) m,
        // the subtraction does not overflow and thus can be non-modular.
        //
        let new_seed =
            (powmod(a, n, m) * self.last as u64) % m +
            (powmod(a, n, (a-1) * m) - 1) / (a-1) * b;

        Rng { last: (new_seed % m) as u32 }
    }
}


// This will end up keeping track of threads, like
// in the other multithreaded Rust version, in
// order to keep writes in order.
//
// This is stolen from another multithreaded Rust
// implementation, although that implementation
// was not able to parallelize the RNG itself.
struct BlockSubmitter<W: io::Write> {
    writer: W,
    pub waiting_on: usize,
}

impl<W: io::Write> BlockSubmitter<W> {
    fn submit(&mut self, data: &[u8], block_num: usize) -> Option<io::Result<()>> {
        if block_num == self.waiting_on {
            self.waiting_on += 1;
            Some(self.submit_async(data))
        }
        else {
            None
        }
    }

    fn submit_async(&mut self, data: &[u8]) -> io::Result<()> {
        self.writer.write_all(data)
    }
}


// For repeating strings as output
fn fasta_static<W: io::Write>(
    writer: &mut W,
    header: &[u8],
    data: &[u8],
    mut n: usize
) -> io::Result<()>
{
    // The aim here is to print a short(ish) string cyclically
    // with line breaks as appropriate.
    //
    // The secret technique is to repeat the string such that
    // any wanted line is a single offset in the string.
    //
    // This technique is stolen from the Haskell version.

    try!(writer.write_all(header));

    // Maximum offset is data.len(),
    // Maximum read len is LINE_LEN
    let stream = data.iter().cloned().cycle();
    let mut extended: Vec<u8> = stream.take(data.len() + LINE_LEN + 1).collect();

    let mut offset = 0;
    while n > 0 {
        let write_len = min(LINE_LEN, n);
        let end = offset + write_len;
        n -= write_len;

        let tmp = extended[end];
        extended[end] = b'\n';
        try!(writer.write_all(&extended[offset..end + 1]));
        extended[end] = tmp;

        offset = end;
        offset %= data.len();
    }

    Ok(())
}


// For RNG streams as output
fn fasta<W: io::Write + Send + 'static>(
    submitter: &Arc<Mutex<BlockSubmitter<W>>>,
    header: &[u8],
    table: &[(u8, f32)],
    rng: &mut Rng,
    n: usize
) -> io::Result<()>
{
    // There's another secret technique in use here:
    // we generate a lookup table to cache search of the
    // aa buffer.
    //
    // The secret technique used is stolen from Haskell's
    // implementation, and is the main secret to the Haskell
    // implementation's  speed.
    fn gen_lookup_table(aa: &[(u8, f32)]) -> Vec<u8> {
        let mut table = Vec::with_capacity(Rng::max_value() as usize + 1);

        let mut cumulative_prob = 0.0;
        let mut cumulative_norm = 0;

        for &(byte, prob) in aa {
            let last_norm = cumulative_norm;
            cumulative_prob += prob;
            cumulative_norm = min(Rng::max_value(), Rng::normalize(cumulative_prob)) + 1;

            table.extend((0..cumulative_norm - last_norm).map(|_| byte));
        }

        table
    }

    {
        try!(submitter.lock().unwrap().submit_async(header));
    }

    let lookup_table = Arc::new(gen_lookup_table(table));

    let thread_count = 4; // avoid external dependency
    let mut threads = Vec::new();
    for block_num in (0..thread_count) {
        let offset = BLOCK_THOROUGHPUT * block_num;

        let local_submitter = submitter.clone();
        let local_lookup_table = lookup_table.clone();
        let local_rng = rng.future(offset as u32);

        threads.push(thread::spawn(move || {
            gen_block(
                local_submitter,
                local_lookup_table,
                local_rng,
                n.saturating_sub(offset),
                block_num,
                thread_count
            )
        }));
    }

    for thread in threads {
        try!(thread.join().unwrap());
    }

    *rng = rng.future(n as u32);

    Ok(())
}

// A very optimized writer.
// I have a feeling a simpler version wouldn't slow
// things down too much, though, since the RNG
// is the really heavy hitter.
fn gen_block<W: io::Write>(
    submitter: Arc<Mutex<BlockSubmitter<W>>>,
    lookup_table: Arc<Vec<u8>>,
    mut rng: Rng,
    mut length: usize,
    mut block_num: usize,
    block_stride: usize,
) -> io::Result<()>
{
    // Include newlines in block
    length += length / LINE_LEN;
    let block: &mut [u8] = &mut [b'\n'; BLOCK_LEN];

    while length > 0 {
        {
            let gen_into = &mut block[..min(length, BLOCK_LEN)];

            // Write random numbers, skipping newlines
            for (i, byte) in gen_into.iter_mut().enumerate() {
                if (i + 1) % (LINE_LEN + 1) != 0 {
                    *byte = lookup_table[rng.gen() as usize];
                }
            }
        }

        let write_out = {
            if length >= BLOCK_LEN               { &mut *block }
            else if length % (LINE_LEN + 1) == 0 { &mut block[..length] }
            else                                 { &mut block[..length + 1] }
        };

        *write_out.last_mut().unwrap() = b'\n';
        loop {
            match submitter.lock().unwrap().submit(write_out, block_num) {
                Some(result) => { try!(result); break; }
                None => std::thread::yield_now()
            }
        }
        block_num += block_stride;
        rng = rng.future((BLOCK_THOROUGHPUT * (block_stride - 1)) as u32);
        length = length.saturating_sub(BLOCK_LEN * (block_stride - 1));

        length = length.saturating_sub(BLOCK_LEN);
    }

    Ok(())
}


fn run<W: io::Write + Send + 'static>(writer: W) -> io::Result<()> {
    let n = std::env::args_os().nth(1)
        .and_then(|s| s.into_string().ok())
        .and_then(|n| n.parse().ok())
        .unwrap_or(1000);

    let rng = &mut Rng::new();

    // Use automatic buffering for the static version...
    let mut writer = io::BufWriter::with_capacity(STDIN_BUF, writer);
    try!(fasta_static(&mut writer, b">ONE Homo sapiens alu\n", ALU, n * 2));

    // ...but the dynamic version does its own buffering already
    let writer = try!(writer.into_inner());
    let submitter = Arc::new(Mutex::new(BlockSubmitter { writer: writer, waiting_on: 0 }));

    { submitter.lock().unwrap().waiting_on = 0; }
    try!(fasta(&submitter, b">TWO IUB ambiguity codes\n", &IUB, rng, n * 3));
    { submitter.lock().unwrap().waiting_on = 0; }
    try!(fasta(&submitter, b">THREE Homo sapiens frequency\n", &HOMOSAPIENS, rng, n * 5));

    Ok(())
}


fn main() {
    run(io::stdout()).unwrap()
}
