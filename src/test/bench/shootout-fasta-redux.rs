// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test reading from os::args()[1] - bogus!

use std::cast::transmute;
use std::from_str::FromStr;
use std::libc::{FILE, STDOUT_FILENO, c_int, fdopen, fputc, fputs, fwrite, size_t};
use std::os;
use std::uint::min;
use std::vec::bytes::copy_memory;
use std::vec;

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

static MESSAGE_1: &'static str = ">ONE Homo sapiens alu\n";
static MESSAGE_2: &'static str = ">TWO IUB ambiguity codes\n";
static MESSAGE_3: &'static str = ">THREE Homo sapiens frequency\n";

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
fn sum_and_scale(a: &'static [AminoAcid]) -> ~[AminoAcid] {
    let mut result = ~[];
    let mut p = 0f32;
    for a_i in a.iter() {
        let mut a_i = *a_i;
        p += a_i.p;
        a_i.p = p * LOOKUP_SCALE;
        result.push(a_i);
    }
    result[result.len() - 1].p = LOOKUP_SCALE;
    result
}

struct AminoAcid {
    c: u8,
    p: f32,
}

struct RepeatFasta {
    alu: &'static str,
    stdout: *FILE,
}

impl RepeatFasta {
    fn new(stdout: *FILE, alu: &'static str) -> RepeatFasta {
        RepeatFasta {
            alu: alu,
            stdout: stdout,
        }
    }

    fn make(&mut self, n: uint) {
        unsafe {
            let stdout = self.stdout;
            let alu_len = self.alu.len();
            let mut buf = vec::from_elem(alu_len + LINE_LEN, 0u8);
            let alu: &[u8] = self.alu.as_bytes();

            copy_memory(buf, alu);
            let buf_len = buf.len();
            copy_memory(buf.mut_slice(alu_len, buf_len),
                        alu.slice_to(LINE_LEN));

            let mut pos = 0;
            let mut bytes;
            let mut n = n;
            while n > 0 {
                bytes = min(LINE_LEN, n);
                fwrite(transmute(&buf[pos]), bytes as size_t, 1, stdout);
                fputc('\n' as c_int, stdout);
                pos += bytes;
                if pos > alu_len {
                    pos -= alu_len;
                }
                n -= bytes;
            }
        }
    }
}

struct RandomFasta {
    seed: u32,
    stdout: *FILE,
    lookup: [AminoAcid, ..LOOKUP_SIZE],
}

impl RandomFasta {
    fn new(stdout: *FILE, a: &[AminoAcid]) -> RandomFasta {
        RandomFasta {
            seed: 42,
            stdout: stdout,
            lookup: RandomFasta::make_lookup(a),
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

    fn make(&mut self, n: uint) {
        unsafe {
            let lines = n / LINE_LEN;
            let chars_left = n % LINE_LEN;
            let mut buf = [0, ..LINE_LEN + 1];

            for _ in range(0, lines) {
                for i in range(0u, LINE_LEN) {
                    buf[i] = self.nextc();
                }
                buf[LINE_LEN] = '\n' as u8;
                fwrite(transmute(&buf[0]),
                       LINE_LEN as size_t + 1,
                       1,
                       self.stdout);
            }
            for i in range(0u, chars_left) {
                buf[i] = self.nextc();
            }
            fwrite(transmute(&buf[0]), chars_left as size_t, 1, self.stdout);
        }
    }
}

fn main() {
    let n: uint = FromStr::from_str(os::args()[1]).unwrap();

    unsafe {
        let mode = "w";
        let stdout = fdopen(STDOUT_FILENO as c_int, transmute(&mode[0]));

        fputs(transmute(&MESSAGE_1[0]), stdout);
        let mut repeat = RepeatFasta::new(stdout, ALU);
        repeat.make(n * 2);

        fputs(transmute(&MESSAGE_2[0]), stdout);
        let iub = sum_and_scale(IUB);
        let mut random = RandomFasta::new(stdout, iub);
        random.make(n * 3);

        fputs(transmute(&MESSAGE_3[0]), stdout);
        let homo_sapiens = sum_and_scale(HOMO_SAPIENS);
        random.lookup = RandomFasta::make_lookup(homo_sapiens);
        random.make(n * 5);

        fputc('\n' as c_int, stdout);
    }
}
