// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![allow(non_snake_case_functions)]

use std::rand::{Rng, task_rng};
use stdtest::Bencher;

use regex::{Regex, NoExpand};

fn bench_assert_match(b: &mut Bencher, re: Regex, text: &str) {
    b.iter(|| if !re.is_match(text) { fail!("no match") });
}

#[bench]
fn no_exponential(b: &mut Bencher) {
    let n = 100;
    let re = Regex::new(format!("{}{}",
                                "a?".repeat(n),
                                "a".repeat(n)).as_slice()).unwrap();
    let text = "a".repeat(n);
    bench_assert_match(b, re, text.as_slice());
}

#[bench]
fn literal(b: &mut Bencher) {
    let re = regex!("y");
    let text = format!("{}y", "x".repeat(50));
    bench_assert_match(b, re, text.as_slice());
}

#[bench]
fn not_literal(b: &mut Bencher) {
    let re = regex!(".y");
    let text = format!("{}y", "x".repeat(50));
    bench_assert_match(b, re, text.as_slice());
}

#[bench]
fn match_class(b: &mut Bencher) {
    let re = regex!("[abcdw]");
    let text = format!("{}w", "xxxx".repeat(20));
    bench_assert_match(b, re, text.as_slice());
}

#[bench]
fn match_class_in_range(b: &mut Bencher) {
    // 'b' is between 'a' and 'c', so the class range checking doesn't help.
    let re = regex!("[ac]");
    let text = format!("{}c", "bbbb".repeat(20));
    bench_assert_match(b, re, text.as_slice());
}

#[bench]
fn replace_all(b: &mut Bencher) {
    let re = regex!("[cjrw]");
    let text = "abcdefghijklmnopqrstuvwxyz";
    // FIXME: This isn't using the $name expand stuff.
    // It's possible RE2/Go is using it, but currently, the expand in this
    // crate is actually compiling a regex, so it's incredibly slow.
    b.iter(|| re.replace_all(text, NoExpand("")));
}

#[bench]
fn anchored_literal_short_non_match(b: &mut Bencher) {
    let re = regex!("^zbc(d|e)");
    let text = "abcdefghijklmnopqrstuvwxyz";
    b.iter(|| re.is_match(text));
}

#[bench]
fn anchored_literal_long_non_match(b: &mut Bencher) {
    let re = regex!("^zbc(d|e)");
    let text = "abcdefghijklmnopqrstuvwxyz".repeat(15);
    b.iter(|| re.is_match(text.as_slice()));
}

#[bench]
fn anchored_literal_short_match(b: &mut Bencher) {
    let re = regex!("^.bc(d|e)");
    let text = "abcdefghijklmnopqrstuvwxyz";
    b.iter(|| re.is_match(text));
}

#[bench]
fn anchored_literal_long_match(b: &mut Bencher) {
    let re = regex!("^.bc(d|e)");
    let text = "abcdefghijklmnopqrstuvwxyz".repeat(15);
    b.iter(|| re.is_match(text.as_slice()));
}

#[bench]
fn one_pass_short_a(b: &mut Bencher) {
    let re = regex!("^.bc(d|e)*$");
    let text = "abcddddddeeeededd";
    b.iter(|| re.is_match(text));
}

#[bench]
fn one_pass_short_a_not(b: &mut Bencher) {
    let re = regex!(".bc(d|e)*$");
    let text = "abcddddddeeeededd";
    b.iter(|| re.is_match(text));
}

#[bench]
fn one_pass_short_b(b: &mut Bencher) {
    let re = regex!("^.bc(?:d|e)*$");
    let text = "abcddddddeeeededd";
    b.iter(|| re.is_match(text));
}

#[bench]
fn one_pass_short_b_not(b: &mut Bencher) {
    let re = regex!(".bc(?:d|e)*$");
    let text = "abcddddddeeeededd";
    b.iter(|| re.is_match(text));
}

#[bench]
fn one_pass_long_prefix(b: &mut Bencher) {
    let re = regex!("^abcdefghijklmnopqrstuvwxyz.*$");
    let text = "abcdefghijklmnopqrstuvwxyz";
    b.iter(|| re.is_match(text));
}

#[bench]
fn one_pass_long_prefix_not(b: &mut Bencher) {
    let re = regex!("^.bcdefghijklmnopqrstuvwxyz.*$");
    let text = "abcdefghijklmnopqrstuvwxyz";
    b.iter(|| re.is_match(text));
}

macro_rules! throughput(
    ($name:ident, $regex:expr, $size:expr) => (
        #[bench]
        fn $name(b: &mut Bencher) {
            let text = gen_text($size);
            b.bytes = $size;
            b.iter(|| if $regex.is_match(text.as_slice()) { fail!("match") });
        }
    );
)

fn easy0() -> Regex { regex!("ABCDEFGHIJKLMNOPQRSTUVWXYZ$") }
fn easy1() -> Regex { regex!("A[AB]B[BC]C[CD]D[DE]E[EF]F[FG]G[GH]H[HI]I[IJ]J$") }
fn medium() -> Regex { regex!("[XYZ]ABCDEFGHIJKLMNOPQRSTUVWXYZ$") }
fn hard() -> Regex { regex!("[ -~]*ABCDEFGHIJKLMNOPQRSTUVWXYZ$") }

fn gen_text(n: uint) -> String {
    let mut rng = task_rng();
    let mut bytes = rng.gen_ascii_chars().map(|n| n as u8).take(n)
                       .collect::<Vec<u8>>();
    for (i, b) in bytes.mut_iter().enumerate() {
        if i % 20 == 0 {
            *b = '\n' as u8
        }
    }
    String::from_utf8(bytes).unwrap()
}

throughput!(easy0_32, easy0(), 32)
throughput!(easy0_1K, easy0(), 1<<10)
throughput!(easy0_32K, easy0(), 32<<10)

throughput!(easy1_32, easy1(), 32)
throughput!(easy1_1K, easy1(), 1<<10)
throughput!(easy1_32K, easy1(), 32<<10)

throughput!(medium_32, medium(), 32)
throughput!(medium_1K, medium(), 1<<10)
throughput!(medium_32K,medium(), 32<<10)

throughput!(hard_32, hard(), 32)
throughput!(hard_1K, hard(), 1<<10)
throughput!(hard_32K,hard(), 32<<10)
