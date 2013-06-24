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
extern mod extra;

use std::int;
use std::io;
use std::os;
use std::rand::Rng;
use std::rand;
use std::result;
use std::str;
use std::uint;

static LINE_LENGTH: uint = 60u;

struct MyRandom {
    last: u32
}

fn myrandom_next(r: @mut MyRandom, mx: u32) -> u32 {
    r.last = (r.last * 3877u32 + 29573u32) % 139968u32;
    mx * r.last / 139968u32
}

struct AminoAcids {
    ch: char,
    prob: u32
}

fn make_cumulative(aa: ~[AminoAcids]) -> ~[AminoAcids] {
    let mut cp: u32 = 0u32;
    let mut ans: ~[AminoAcids] = ~[];
    for aa.iter().advance |a| {
        cp += a.prob;
        ans += [AminoAcids {ch: a.ch, prob: cp}];
    }
    ans
}

fn select_random(r: u32, genelist: ~[AminoAcids]) -> char {
    if r < genelist[0].prob { return genelist[0].ch; }
    fn bisect(v: ~[AminoAcids], lo: uint, hi: uint, target: u32) -> char {
        if hi > lo + 1u {
            let mid: uint = lo + (hi - lo) / 2u;
            if target < v[mid].prob {
                return bisect(v, lo, mid, target);
            } else { return bisect(v, mid, hi, target); }
        } else { return v[hi].ch; }
    }
    bisect(copy genelist, 0, genelist.len() - 1, r)
}

fn make_random_fasta(wr: @io::Writer,
                     id: ~str,
                     desc: ~str,
                     genelist: ~[AminoAcids],
                     n: int) {
    wr.write_line(~">" + id + " " + desc);
    let mut rng = rand::rng();
    let rng = @mut MyRandom {
        last: rng.next()
    };
    let mut op: ~str = ~"";
    for uint::range(0u, n as uint) |_i| {
        op.push_char(select_random(myrandom_next(rng, 100u32),
                                              copy genelist));
        if op.len() >= LINE_LENGTH {
            wr.write_line(op);
            op = ~"";
        }
    }
    if op.len() > 0u { wr.write_line(op); }
}

fn make_repeat_fasta(wr: @io::Writer, id: ~str, desc: ~str, s: ~str, n: int) {
    wr.write_line(~">" + id + " " + desc);
    let mut op = str::with_capacity( LINE_LENGTH );
    let sl = s.len();
    for uint::range(0u, n as uint) |i| {
        if (op.len() >= LINE_LENGTH) {
            wr.write_line( op );
            op = str::with_capacity( LINE_LENGTH );
        }
        op.push_char( s[i % sl] as char );
    }
    if op.len() > 0 {
        wr.write_line(op)
    }
}

fn acid(ch: char, prob: u32) -> AminoAcids {
    AminoAcids {ch: ch, prob: prob}
}

fn main() {
    let args = os::args();
    let args = if os::getenv("RUST_BENCH").is_some() {
        // alioth tests k-nucleotide with this data at 25,000,000
        ~[~"", ~"5000000"]
    } else if args.len() <= 1u {
        ~[~"", ~"1000"]
    } else {
        args
    };

    let writer = if os::getenv("RUST_BENCH").is_some() {
        result::get(&io::file_writer(&Path("./shootout-fasta.data"),
                                    [io::Truncate, io::Create]))
    } else {
        io::stdout()
    };

    let n = int::from_str(args[1]).get();

    let iub: ~[AminoAcids] =
        make_cumulative(~[acid('a', 27u32), acid('c', 12u32), acid('g', 12u32),
                         acid('t', 27u32), acid('B', 2u32), acid('D', 2u32),
                         acid('H', 2u32), acid('K', 2u32), acid('M', 2u32),
                         acid('N', 2u32), acid('R', 2u32), acid('S', 2u32),
                         acid('V', 2u32), acid('W', 2u32), acid('Y', 2u32)]);
    let homosapiens: ~[AminoAcids] =
        make_cumulative(~[acid('a', 30u32), acid('c', 20u32), acid('g', 20u32),
                         acid('t', 30u32)]);
    let alu: ~str =
        ~"GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGG\
          GAGGCCGAGGCGGGCGGATCACCTGAGGTCAGGAGTTCGAGA\
          CCAGCCTGGCCAACATGGTGAAACCCCGTCTCTACTAAAAAT\
          ACAAAAATTAGCCGGGCGTGGTGGCGCGCGCCTGTAATCCCA\
          GCTACTCGGGAGGCTGAGGCAGGAGAATCGCTTGAACCCGGG\
          AGGCGGAGGTTGCAGTGAGCCGAGATCGCGCCACTGCACTCC\
          AGCCTGGGCGACAGAGCGAGACTCCGTCTCAAAAA";
    make_repeat_fasta(writer, ~"ONE", ~"Homo sapiens alu", alu, n * 2);
    make_random_fasta(writer, ~"TWO", ~"IUB ambiguity codes", iub, n * 3);
    make_random_fasta(writer, ~"THREE",
                      ~"Homo sapiens frequency", homosapiens, n * 5);
}
