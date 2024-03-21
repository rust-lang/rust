// The Computer Language Benchmarks Game
// https://benchmarksgame-team.pages.debian.net/benchmarksgame/
//
// contributed by the Rust Project Developers
// contributed by TeXitoi
// contributed by BurntSushi

use std::io::{self, Read};

macro_rules! regex {
    ($re:expr) => {
        ::regex::Regex::new($re).unwrap()
    };
}

fn main() {
    let mut seq = String::with_capacity(50 * (1 << 20));
    io::stdin().read_to_string(&mut seq).unwrap();
    let ilen = seq.len();

    seq = regex!(">[^\n]*\n|\n").replace_all(&seq, "").into_owned();
    let clen = seq.len();

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
    for re in variants {
        println!("{} {}", re.to_string(), re.find_iter(&seq).count());
    }

    let substs = vec![
        (b'B', "(c|g|t)"),
        (b'D', "(a|g|t)"),
        (b'H', "(a|c|t)"),
        (b'K', "(g|t)"),
        (b'M', "(a|c)"),
        (b'N', "(a|c|g|t)"),
        (b'R', "(a|g)"),
        (b'S', "(c|g)"),
        (b'V', "(a|c|g)"),
        (b'W', "(a|t)"),
        (b'Y', "(c|t)"),
    ]; // combined into one regex in `replace_all`
    let seq = replace_all(&seq, substs);

    println!("\n{}\n{}\n{}", ilen, clen, seq.len());
}

fn replace_all(text: &str, substs: Vec<(u8, &str)>) -> String {
    let mut replacements = vec![""; 256];
    let mut alternates = vec![];
    for (re, replacement) in substs {
        replacements[re as usize] = replacement;
        alternates.push((re as char).to_string());
    }

    let re = regex!(&alternates.join("|"));
    let mut new = String::with_capacity(text.len());
    let mut last_match = 0;
    for m in re.find_iter(text) {
        new.push_str(&text[last_match..m.start()]);
        new.push_str(replacements[text.as_bytes()[m.start()] as usize]);
        last_match = m.end();
    }
    new.push_str(&text[last_match..]);
    new
}
