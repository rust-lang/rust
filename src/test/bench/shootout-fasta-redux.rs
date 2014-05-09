// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp::min;
use std::io::{stdout, IoResult};
use std::os;
use std::slice::bytes::copy_memory;

static LINE_LEN: uint = 60;
static LOOKUP_SIZE: uint = 4 * 1024;
static LOOKUP_SCALE: f32 = (LOOKUP_SIZE - 1) as f32;

// Random number generator constants
static IM: u32 = 139968;
static IA: u32 = 3877;
static IC: u32 = 29573;

static ALU: &'static str = "GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTG\
                            GGAGGCCGAGGCGGGCGGATCACCTGAGGTCAGGAGTTCGA\
                            GACCAGCCTGGCCAACATGGTGAAACCCCGTCTCTACTAAA\
                            AATACAAAAATTAGCCGGGCGTGGTGGCGCGCGCCTGTAAT\
                            CCCAGCTACTCGGGAGGCTGAGGCAGGAGAATCGCTTGAAC\
                            CCGGGAGGCGGAGGTTGCAGTGAGCCGAGATCGCGCCACTG\
                            CACTCCAGCCTGGGCGACAGAGCGAGACTCCGTCTCAAAAA";

static NULL_AMINO_ACID: AminoAcid = AminoAcid { c: ' ' as u8, p: 0.0 };

static IUB: [AminoAcid, ..15] = [
    AminoAcid { c: 'a' as u8, p: 0.27 },
    AminoAcid { c: 'c' as u8, p: 0.12 },
    AminoAcid { c: 'g' as u8, p: 0.12 },
    AminoAcid { c: 't' as u8, p: 0.27 },
    AminoAcid { c: 'B' as u8, p: 0.02 },
    AminoAcid { c: 'D' as u8, p: 0.02 },
    AminoAcid { c: 'H' as u8, p: 0.02 },
    AminoAcid { c: 'K' as u8, p: 0.02 },
    AminoAcid { c: 'M' as u8, p: 0.02 },
    AminoAcid { c: 'N' as u8, p: 0.02 },
    AminoAcid { c: 'R' as u8, p: 0.02 },
    AminoAcid { c: 'S' as u8, p: 0.02 },
    AminoAcid { c: 'V' as u8, p: 0.02 },
    AminoAcid { c: 'W' as u8, p: 0.02 },
    AminoAcid { c: 'Y' as u8, p: 0.02 },
];

static HOMO_SAPIENS: [AminoAcid, ..4] = [
    AminoAcid { c: 'a' as u8, p: 0.3029549426680 },
    AminoAcid { c: 'c' as u8, p: 0.1979883004921 },
    AminoAcid { c: 'g' as u8, p: 0.1975473066391 },
    AminoAcid { c: 't' as u8, p: 0.3015094502008 },
];

// FIXME: Use map().
fn sum_and_scale(a: &'static [AminoAcid]) -> Vec<AminoAcid> {
    let mut result = Vec::new();
    let mut p = 0f32;
    for a_i in a.iter() {
        let mut a_i = *a_i;
        p += a_i.p;
        a_i.p = p * LOOKUP_SCALE;
        result.push(a_i);
    }
    let result_len = result.len();
    result.get_mut(result_len - 1).p = LOOKUP_SCALE;
    result
}

struct AminoAcid {
    c: u8,
    p: f32,
}

struct RepeatFasta<'a, W> {
    alu: &'static str,
    out: &'a mut W
}

impl<'a, W: Writer> RepeatFasta<'a, W> {
    fn new(alu: &'static str, w: &'a mut W) -> RepeatFasta<'a, W> {
        RepeatFasta { alu: alu, out: w }
    }

    fn make(&mut self, n: uint) -> IoResult<()> {
        let alu_len = self.alu.len();
        let mut buf = Vec::from_elem(alu_len + LINE_LEN, 0u8);
        let alu: &[u8] = self.alu.as_bytes();

        copy_memory(buf.as_mut_slice(), alu);
        let buf_len = buf.len();
        copy_memory(buf.mut_slice(alu_len, buf_len),
                    alu.slice_to(LINE_LEN));

        let mut pos = 0;
        let mut bytes;
        let mut n = n;
        while n > 0 {
            bytes = min(LINE_LEN, n);
            try!(self.out.write(buf.slice(pos, pos + bytes)));
            try!(self.out.write_u8('\n' as u8));
            pos += bytes;
            if pos > alu_len {
                pos -= alu_len;
            }
            n -= bytes;
        }
        Ok(())
    }
}

fn make_lookup(a: &[AminoAcid]) -> [AminoAcid, ..LOOKUP_SIZE] {
    let mut lookup = [ NULL_AMINO_ACID, ..LOOKUP_SIZE ];
    let mut j = 0;
    for (i, slot) in lookup.mut_iter().enumerate() {
        while a[j].p < (i as f32) {
            j += 1;
        }
        *slot = a[j];
    }
    lookup
}

struct RandomFasta<'a, W> {
    seed: u32,
    lookup: [AminoAcid, ..LOOKUP_SIZE],
    out: &'a mut W,
}

impl<'a, W: Writer> RandomFasta<'a, W> {
    fn new(w: &'a mut W, a: &[AminoAcid]) -> RandomFasta<'a, W> {
        RandomFasta {
            seed: 42,
            out: w,
            lookup: make_lookup(a),
        }
    }

    fn rng(&mut self, max: f32) -> f32 {
        self.seed = (self.seed * IA + IC) % IM;
        max * (self.seed as f32) / (IM as f32)
    }

    fn nextc(&mut self) -> u8 {
        let r = self.rng(1.0);
        for a in self.lookup.iter() {
            if a.p >= r {
                return a.c;
            }
        }
        0
    }

    fn make(&mut self, n: uint) -> IoResult<()> {
        let lines = n / LINE_LEN;
        let chars_left = n % LINE_LEN;
        let mut buf = [0, ..LINE_LEN + 1];

        for _ in range(0, lines) {
            for i in range(0u, LINE_LEN) {
                buf[i] = self.nextc();
            }
            buf[LINE_LEN] = '\n' as u8;
            try!(self.out.write(buf));
        }
        for i in range(0u, chars_left) {
            buf[i] = self.nextc();
        }
        self.out.write(buf.slice_to(chars_left))
    }
}

fn main() {
    let args = os::args();
    let args = args.as_slice();
    let n = if args.len() > 1 {
        from_str::<uint>(args[1]).unwrap()
    } else {
        5
    };

    let mut out = stdout();

    out.write_line(">ONE Homo sapiens alu").unwrap();
    {
        let mut repeat = RepeatFasta::new(ALU, &mut out);
        repeat.make(n * 2).unwrap();
    }

    out.write_line(">TWO IUB ambiguity codes").unwrap();
    let iub = sum_and_scale(IUB);
    let mut random = RandomFasta::new(&mut out, iub.as_slice());
    random.make(n * 3).unwrap();

    random.out.write_line(">THREE Homo sapiens frequency").unwrap();
    let homo_sapiens = sum_and_scale(HOMO_SAPIENS);
    random.lookup = make_lookup(homo_sapiens.as_slice());
    random.make(n * 5).unwrap();

    random.out.write_str("\n").unwrap();
}
