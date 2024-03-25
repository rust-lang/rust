// The Computer Language Benchmarks Game
// https://benchmarksgame-team.pages.debian.net/benchmarksgame/
//
// contributed by the Rust Project Developers
// contributed by TeXitoi
// contributed by BurntSushi

use std::io::{self, Read};
use std::sync::Arc;
use std::thread;

macro_rules! regex {
    ($re:expr) => {
        ::regex::bytes::Regex::new($re).unwrap()
    };
}

fn main() {
    let mut seq = Vec::with_capacity(51 * (1 << 20));
    io::stdin().read_to_end(&mut seq).unwrap();
    let ilen = seq.len();

    seq = regex!(">[^\n]*\n|\n").replace_all(&seq, &b""[..]).into_owned();
    let clen = seq.len();
    let seq_arc = Arc::new(seq.clone());

    let variants = vec![
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
    let mut counts = vec![];
    for variant in variants {
        let seq = seq_arc.clone();
        let restr = variant.to_string();
        let future = thread::spawn(move || variant.find_iter(&seq).count());
        counts.push((restr, future));
    }

    let substs = vec![
        (regex!("B"), &b"(c|g|t)"[..]),
        (regex!("D"), &b"(a|g|t)"[..]),
        (regex!("H"), &b"(a|c|t)"[..]),
        (regex!("K"), &b"(g|t)"[..]),
        (regex!("M"), &b"(a|c)"[..]),
        (regex!("N"), &b"(a|c|g|t)"[..]),
        (regex!("R"), &b"(a|g)"[..]),
        (regex!("S"), &b"(c|g)"[..]),
        (regex!("V"), &b"(a|c|g)"[..]),
        (regex!("W"), &b"(a|t)"[..]),
        (regex!("Y"), &b"(c|t)"[..]),
    ];
    let mut seq = seq;
    for (re, replacement) in substs {
        seq = re.replace_all(&seq, replacement).into_owned();
    }

    for (variant, count) in counts {
        println!("{} {}", variant, count.join().unwrap());
    }
    println!("\n{}\n{}\n{}", ilen, clen, seq.len());
}
