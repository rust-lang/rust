

/* -*- mode: rust; indent-tabs-mode: nil -*-
 * Implementation of 'fasta' benchmark from
 * Computer Language Benchmarks Game
 * http://shootout.alioth.debian.org/
 */
use std;
import std::vec;
import std::str;
import std::uint;
import std::int;

fn LINE_LENGTH() -> uint { ret 60u; }

obj myrandom(mutable u32 last) {
    fn next(u32 mx) -> u32 {
        last = (last * 3877u32 + 29573u32) % 139968u32;
        auto ans = mx * last / 139968u32;
        ret ans;
    }
}

type aminoacids = tup(char, u32);

fn make_cumulative(vec[aminoacids] aa) -> vec[aminoacids] {
    let u32 cp = 0u32;
    let vec[aminoacids] ans = [];
    for (aminoacids a in aa) { cp += a._1; ans += [tup(a._0, cp)]; }
    ret ans;
}

fn select_random(u32 r, vec[aminoacids] genelist) -> char {
    if (r < genelist.(0)._1) { ret genelist.(0)._0; }
    fn bisect(vec[aminoacids] v, uint lo, uint hi, u32 target) -> char {
        if (hi > lo + 1u) {
            let uint mid = lo + (hi - lo) / 2u;
            if (target < v.(mid)._1) {
                be bisect(v, lo, mid, target);
            } else { be bisect(v, mid, hi, target); }
        } else { ret v.(hi)._0; }
    }
    ret bisect(genelist, 0u, vec::len[aminoacids](genelist) - 1u, r);
}

fn make_random_fasta(str id, str desc, vec[aminoacids] genelist, int n) {
    log ">" + id + " " + desc;
    auto rng = myrandom(std::rand::mk_rng().next());
    let str op = "";
    for each (uint i in uint::range(0u, n as uint)) {
        str::push_byte(op, select_random(rng.next(100u32), genelist) as u8);
        if (str::byte_len(op) >= LINE_LENGTH()) { log op; op = ""; }
    }
    if (str::byte_len(op) > 0u) { log op; }
}

fn make_repeat_fasta(str id, str desc, str s, int n) {
    log ">" + id + " " + desc;
    let str op = "";
    let uint sl = str::byte_len(s);
    for each (uint i in uint::range(0u, n as uint)) {
        str::push_byte(op, s.(i % sl));
        if (str::byte_len(op) >= LINE_LENGTH()) { log op; op = ""; }
    }
    if (str::byte_len(op) > 0u) { log op; }
}

fn main(vec[str] args) {
    let vec[aminoacids] iub =
        make_cumulative([tup('a', 27u32), tup('c', 12u32), tup('g', 12u32),
                         tup('t', 27u32), tup('B', 2u32), tup('D', 2u32),
                         tup('H', 2u32), tup('K', 2u32), tup('M', 2u32),
                         tup('N', 2u32), tup('R', 2u32), tup('S', 2u32),
                         tup('V', 2u32), tup('W', 2u32), tup('Y', 2u32)]);
    let vec[aminoacids] homosapiens =
        make_cumulative([tup('a', 30u32), tup('c', 20u32), tup('g', 20u32),
                         tup('t', 30u32)]);
    let str alu =
        "GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGG" +
            "GAGGCCGAGGCGGGCGGATCACCTGAGGTCAGGAGTTCGAGA" +
            "CCAGCCTGGCCAACATGGTGAAACCCCGTCTCTACTAAAAAT" +
            "ACAAAAATTAGCCGGGCGTGGTGGCGCGCGCCTGTAATCCCA" +
            "GCTACTCGGGAGGCTGAGGCAGGAGAATCGCTTGAACCCGGG" +
            "AGGCGGAGGTTGCAGTGAGCCGAGATCGCGCCACTGCACTCC" +
            "AGCCTGGGCGACAGAGCGAGACTCCGTCTCAAAAA";
    let int n = 512;
    make_repeat_fasta("ONE", "Homo sapiens alu", alu, n * 2);
    make_random_fasta("TWO", "IUB ambiguity codes", iub, n * 3);
    make_random_fasta("THREE", "Homo sapiens frequency", homosapiens, n * 5);
}