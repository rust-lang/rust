

/* -*- mode: rust; indent-tabs-mode: nil -*-
 * Implementation of 'fasta' benchmark from
 * Computer Language Benchmarks Game
 * http://shootout.alioth.debian.org/
 */
use std;
import vec;
import uint;
import int;
import str;
import io::writer_util;

fn LINE_LENGTH() -> uint { ret 60u; }

type myrandom = @{mut last: u32};

fn myrandom_next(r: myrandom, mx: u32) -> u32 {
    r.last = (r.last * 3877u32 + 29573u32) % 139968u32;
    mx * r.last / 139968u32
}

type aminoacids = {ch: char, prob: u32};

fn make_cumulative(aa: ~[aminoacids]) -> ~[aminoacids] {
    let mut cp: u32 = 0u32;
    let mut ans: ~[aminoacids] = ~[];
    for aa.each |a| { cp += a.prob; ans += ~[{ch: a.ch, prob: cp}]; }
    ret ans;
}

fn select_random(r: u32, genelist: ~[aminoacids]) -> char {
    if r < genelist[0].prob { ret genelist[0].ch; }
    fn bisect(v: ~[aminoacids], lo: uint, hi: uint, target: u32) -> char {
        if hi > lo + 1u {
            let mid: uint = lo + (hi - lo) / 2u;
            if target < v[mid].prob {
                ret bisect(v, lo, mid, target);
            } else { ret bisect(v, mid, hi, target); }
        } else { ret v[hi].ch; }
    }
    ret bisect(genelist, 0u, vec::len::<aminoacids>(genelist) - 1u, r);
}

fn make_random_fasta(wr: io::writer, id: str, desc: str, genelist: ~[aminoacids], n: int) {
    wr.write_line(">" + id + " " + desc);
    let rng = @{mut last: rand::rng().next()};
    let mut op: str = "";
    for uint::range(0u, n as uint) |_i| {
        str::push_char(op, select_random(myrandom_next(rng, 100u32),
                                         genelist));
        if str::len(op) >= LINE_LENGTH() {
            wr.write_line(op);
            op = "";
        }
    }
    if str::len(op) > 0u { wr.write_line(op); }
}

fn make_repeat_fasta(wr: io::writer, id: str, desc: str, s: str, n: int) unsafe {
    wr.write_line(">" + id + " " + desc);
    let mut op: str = "";
    let sl: uint = str::len(s);
    for uint::range(0u, n as uint) |i| {
        str::unsafe::push_byte(op, s[i % sl]);
        if str::len(op) >= LINE_LENGTH() {
            wr.write_line(op);
            op = "";
        }
    }
    if str::len(op) > 0u { wr.write_line(op); }
}

fn acid(ch: char, prob: u32) -> aminoacids { ret {ch: ch, prob: prob}; }

fn main(args: ~[str]) {
    let args = if os::getenv("RUST_BENCH").is_some() {
        // alioth tests k-nucleotide with this data at 25,000,000
        ~["", "5000000"]
    } else if args.len() <= 1u {
        ~["", "1000"]
    } else {
        args
    };

    let writer = if os::getenv("RUST_BENCH").is_some() {
        result::get(io::file_writer("./shootout-fasta.data", ~[io::truncate, io::create]))
    } else {
        io::stdout()
    };

    let n = int::from_str(args[1]).get();

    let iub: ~[aminoacids] =
        make_cumulative(~[acid('a', 27u32), acid('c', 12u32), acid('g', 12u32),
                         acid('t', 27u32), acid('B', 2u32), acid('D', 2u32),
                         acid('H', 2u32), acid('K', 2u32), acid('M', 2u32),
                         acid('N', 2u32), acid('R', 2u32), acid('S', 2u32),
                         acid('V', 2u32), acid('W', 2u32), acid('Y', 2u32)]);
    let homosapiens: ~[aminoacids] =
        make_cumulative(~[acid('a', 30u32), acid('c', 20u32), acid('g', 20u32),
                         acid('t', 30u32)]);
    let alu: str =
        "GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGG" +
            "GAGGCCGAGGCGGGCGGATCACCTGAGGTCAGGAGTTCGAGA" +
            "CCAGCCTGGCCAACATGGTGAAACCCCGTCTCTACTAAAAAT" +
            "ACAAAAATTAGCCGGGCGTGGTGGCGCGCGCCTGTAATCCCA" +
            "GCTACTCGGGAGGCTGAGGCAGGAGAATCGCTTGAACCCGGG" +
            "AGGCGGAGGTTGCAGTGAGCCGAGATCGCGCCACTGCACTCC" +
            "AGCCTGGGCGACAGAGCGAGACTCCGTCTCAAAAA";
    make_repeat_fasta(writer, "ONE", "Homo sapiens alu", alu, n * 2);
    make_random_fasta(writer, "TWO", "IUB ambiguity codes", iub, n * 3);
    make_random_fasta(writer, "THREE",
                      "Homo sapiens frequency", homosapiens, n * 5);
}
