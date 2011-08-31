

/* -*- mode: rust; indent-tabs-mode: nil -*-
 * Implementation of 'fasta' benchmark from
 * Computer Language Benchmarks Game
 * http://shootout.alioth.debian.org/
 */
use std;
import std::vec;
import std::uint;
import std::int;
import std::istr;

fn LINE_LENGTH() -> uint { ret 60u; }

obj myrandom(mutable last: u32) {
    fn next(mx: u32) -> u32 {
        last = (last * 3877u32 + 29573u32) % 139968u32;
        let ans = mx * last / 139968u32;
        ret ans;
    }
}

type aminoacids = {ch: char, prob: u32};

fn make_cumulative(aa: &[aminoacids]) -> [aminoacids] {
    let cp: u32 = 0u32;
    let ans: [aminoacids] = [];
    for a: aminoacids in aa { cp += a.prob; ans += [{ch: a.ch, prob: cp}]; }
    ret ans;
}

fn select_random(r: u32, genelist: &[aminoacids]) -> char {
    if r < genelist[0].prob { ret genelist[0].ch; }
    fn bisect(v: &[aminoacids], lo: uint, hi: uint, target: u32) -> char {
        if hi > lo + 1u {
            let mid: uint = lo + (hi - lo) / 2u;
            if target < v[mid].prob {
                be bisect(v, lo, mid, target);
            } else { be bisect(v, mid, hi, target); }
        } else { ret v[hi].ch; }
    }
    ret bisect(genelist, 0u, vec::len::<aminoacids>(genelist) - 1u, r);
}

fn make_random_fasta(id: &istr, desc: &istr,
                     genelist: &[aminoacids], n: int) {
    log ~">" + id + ~" " + desc;
    let rng = myrandom(std::rand::mk_rng().next());
    let op: istr = ~"";
    for each i: uint in uint::range(0u, n as uint) {
        istr::push_byte(op, select_random(rng.next(100u32), genelist) as u8);
        if istr::byte_len(op) >= LINE_LENGTH() { log op; op = ~""; }
    }
    if istr::byte_len(op) > 0u { log op; }
}

fn make_repeat_fasta(id: &istr, desc: &istr, s: &istr, n: int) {
    log ~">" + id + ~" " + desc;
    let op: istr = ~"";
    let sl: uint = istr::byte_len(s);
    for each i: uint in uint::range(0u, n as uint) {
        istr::push_byte(op, s[i % sl]);
        if istr::byte_len(op) >= LINE_LENGTH() { log op; op = ~""; }
    }
    if istr::byte_len(op) > 0u { log op; }
}

fn acid(ch: char, prob: u32) -> aminoacids { ret {ch: ch, prob: prob}; }

fn main(args: [str]) {
    let iub: [aminoacids] =
        make_cumulative([acid('a', 27u32), acid('c', 12u32), acid('g', 12u32),
                         acid('t', 27u32), acid('B', 2u32), acid('D', 2u32),
                         acid('H', 2u32), acid('K', 2u32), acid('M', 2u32),
                         acid('N', 2u32), acid('R', 2u32), acid('S', 2u32),
                         acid('V', 2u32), acid('W', 2u32), acid('Y', 2u32)]);
    let homosapiens: [aminoacids] =
        make_cumulative([acid('a', 30u32), acid('c', 20u32), acid('g', 20u32),
                         acid('t', 30u32)]);
    let alu: istr =
        ~"GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGG" +
            ~"GAGGCCGAGGCGGGCGGATCACCTGAGGTCAGGAGTTCGAGA" +
            ~"CCAGCCTGGCCAACATGGTGAAACCCCGTCTCTACTAAAAAT" +
            ~"ACAAAAATTAGCCGGGCGTGGTGGCGCGCGCCTGTAATCCCA" +
            ~"GCTACTCGGGAGGCTGAGGCAGGAGAATCGCTTGAACCCGGG" +
            ~"AGGCGGAGGTTGCAGTGAGCCGAGATCGCGCCACTGCACTCC" +
            ~"AGCCTGGGCGACAGAGCGAGACTCCGTCTCAAAAA";
    let n: int = 512;
    make_repeat_fasta(~"ONE", ~"Homo sapiens alu", alu, n * 2);
    make_random_fasta(~"TWO", ~"IUB ambiguity codes", iub, n * 3);
    make_random_fasta(~"THREE", ~"Homo sapiens frequency",
                      homosapiens, n * 5);
}
