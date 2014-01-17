// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/* -*- mode: rust; indent-tabs-mode: nil -*-
 * Implementation of 'fasta' benchmark from
 * Computer Language Benchmarks Game
 * http://shootout.alioth.debian.org/
 */

use std::io;
use std::io::{BufferedWriter, File};
use std::num::min;
use std::os;

static LINE_LENGTH: uint = 60;
static IM: u32 = 139968;

struct MyRandom {
    last: u32
}
impl MyRandom {
    fn new() -> MyRandom { MyRandom { last: 42 } }
    fn normalize(p: f32) -> u32 {(p * IM as f32).floor() as u32}
    fn gen(&mut self) -> u32 {
        self.last = (self.last * 3877 + 29573) % IM;
        self.last
    }
}

struct AAGen<'a> {
    rng: &'a mut MyRandom,
    data: ~[(u32, u8)]
}
impl<'a> AAGen<'a> {
    fn new<'b>(rng: &'b mut MyRandom, aa: &[(char, f32)]) -> AAGen<'b> {
        let mut cum = 0.;
        let data = aa.iter()
            .map(|&(ch, p)| { cum += p; (MyRandom::normalize(cum), ch as u8) })
            .collect();
        AAGen { rng: rng, data: data }
    }
}
impl<'a> Iterator<u8> for AAGen<'a> {
    fn next(&mut self) -> Option<u8> {
        let r = self.rng.gen();
        self.data.iter()
            .skip_while(|pc| pc.n0() < r)
            .map(|&(_, c)| c)
            .next()
    }
}

fn make_fasta<W: Writer, I: Iterator<u8>>(
    wr: &mut W, header: &str, mut it: I, mut n: uint)
{
    wr.write(header.as_bytes());
    let mut line = [0u8, .. LINE_LENGTH + 1];
    while n > 0 {
        let nb = min(LINE_LENGTH, n);
        for i in range(0, nb) {
            line[i] = it.next().unwrap();
        }
        n -= nb;
        line[nb] = '\n' as u8;
        wr.write(line.slice_to(nb + 1));
    }
}

fn run<W: Writer>(writer: &mut W) {
    let args = os::args();
    let n = if os::getenv("RUST_BENCH").is_some() {
        25000000
    } else if args.len() <= 1u {
        1000
    } else {
        from_str(args[1]).unwrap()
    };

    let rng = &mut MyRandom::new();
    let alu =
        "GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGG\
        GAGGCCGAGGCGGGCGGATCACCTGAGGTCAGGAGTTCGAGA\
        CCAGCCTGGCCAACATGGTGAAACCCCGTCTCTACTAAAAAT\
        ACAAAAATTAGCCGGGCGTGGTGGCGCGCGCCTGTAATCCCA\
        GCTACTCGGGAGGCTGAGGCAGGAGAATCGCTTGAACCCGGG\
        AGGCGGAGGTTGCAGTGAGCCGAGATCGCGCCACTGCACTCC\
        AGCCTGGGCGACAGAGCGAGACTCCGTCTCAAAAA";
    let iub = &[('a', 0.27), ('c', 0.12), ('g', 0.12),
                ('t', 0.27), ('B', 0.02), ('D', 0.02),
                ('H', 0.02), ('K', 0.02), ('M', 0.02),
                ('N', 0.02), ('R', 0.02), ('S', 0.02),
                ('V', 0.02), ('W', 0.02), ('Y', 0.02)];
    let homosapiens = &[('a', 0.3029549426680),
                        ('c', 0.1979883004921),
                        ('g', 0.1975473066391),
                        ('t', 0.3015094502008)];

    make_fasta(writer, ">ONE Homo sapiens alu\n",
               alu.as_bytes().iter().cycle().map(|c| *c), n * 2);
    make_fasta(writer, ">TWO IUB ambiguity codes\n",
               AAGen::new(rng, iub), n * 3);
    make_fasta(writer, ">THREE Homo sapiens frequency\n",
               AAGen::new(rng, homosapiens), n * 5);

    writer.flush();
}

fn main() {
    if os::getenv("RUST_BENCH").is_some() {
        let mut file = BufferedWriter::new(File::create(&Path::new("./shootout-fasta.data")));
        run(&mut file);
    } else {
        run(&mut BufferedWriter::new(io::stdout()));
    }
}
