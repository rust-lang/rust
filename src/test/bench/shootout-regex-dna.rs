// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME(#13725) windows needs fixing.
// ignore-win32
// ignore-stage1
// ignore-cross-compile #12102

#![feature(macro_rules, phase)]

extern crate regex;
#[phase(syntax)]extern crate regex_macros;
extern crate sync;

use std::io;
use regex::{NoExpand, Regex};
use sync::Arc;

fn count_matches(seq: &str, variant: &Regex) -> int {
    let mut n = 0;
    for _ in variant.find_iter(seq) {
        n += 1;
    }
    n
}

fn main() {
    let mut rdr = if std::os::getenv("RUST_BENCH").is_some() {
        let fd = io::File::open(&Path::new("shootout-k-nucleotide.data"));
        box io::BufferedReader::new(fd) as Box<io::Reader>
    } else {
        box io::stdin() as Box<io::Reader>
    };
    let mut seq = StrBuf::from_str(rdr.read_to_str().unwrap());
    let ilen = seq.len();

    seq = regex!(">[^\n]*\n|\n").replace_all(seq.as_slice(), NoExpand(""));
    let seq_arc = Arc::new(seq.clone()); // copy before it moves
    let clen = seq.len();

    let mut seqlen = sync::Future::spawn(proc() {
        let substs = ~[
            (regex!("B"), "(c|g|t)"),
            (regex!("D"), "(a|g|t)"),
            (regex!("H"), "(a|c|t)"),
            (regex!("K"), "(g|t)"),
            (regex!("M"), "(a|c)"),
            (regex!("N"), "(a|c|g|t)"),
            (regex!("R"), "(a|g)"),
            (regex!("S"), "(c|g)"),
            (regex!("V"), "(a|c|g)"),
            (regex!("W"), "(a|t)"),
            (regex!("Y"), "(c|t)"),
        ];
        let mut seq = seq;
        for (re, replacement) in substs.move_iter() {
            seq = re.replace_all(seq.as_slice(), NoExpand(replacement));
        }
        seq.len()
    });

    let variants = ~[
        regex!("agggtaaa|tttaccct"),
        regex!("[cgt]gggtaaa|tttaccc[acg]"),
        regex!("a[act]ggtaaa|tttacc[agt]t"),
        regex!("ag[act]gtaaa|tttac[agt]ct"),
        regex!("agg[act]taaa|ttta[agt]cct"),
        regex!("aggg[acg]aaa|ttt[cgt]ccct"),
        regex!("agggt[cgt]aa|tt[acg]accct"),
        regex!("agggta[cgt]a|t[acg]taccct"),
        regex!("agggtaa[cgt]|[acg]ttaccct"),
    ];
    let (mut variant_strs, mut counts) = (vec!(), vec!());
    for variant in variants.move_iter() {
        let seq_arc_copy = seq_arc.clone();
        variant_strs.push(variant.to_str().to_owned());
        counts.push(sync::Future::spawn(proc() {
            count_matches(seq_arc_copy.as_slice(), &variant)
        }));
    }

    for (i, variant) in variant_strs.iter().enumerate() {
        println!("{} {}", variant, counts.get_mut(i).get());
    }
    println!("");
    println!("{}", ilen);
    println!("{}", clen);
    println!("{}", seqlen.get());
}
