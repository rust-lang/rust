#![feature(test)]

extern crate test;
extern crate rustc_serialize;

use test::Bencher;
use rustc_serialize::hex::{FromHex, ToHex};

#[bench]
fn bench_to_hex(b: &mut Bencher) {
    let s = "イロハニホヘト チリヌルヲ ワカヨタレソ ツネナラム \
             ウヰノオクヤマ ケフコエテ アサキユメミシ ヱヒモセスン";
    b.iter(|| {
        s.as_bytes().to_hex();
    });
    b.bytes = s.len() as u64;
}

#[bench]
fn bench_from_hex(b: &mut Bencher) {
    let s = "イロハニホヘト チリヌルヲ ワカヨタレソ ツネナラム \
             ウヰノオクヤマ ケフコエテ アサキユメミシ ヱヒモセスン";
    let sb = s.as_bytes().to_hex();
    b.iter(|| {
        sb.from_hex().unwrap();
    });
    b.bytes = sb.len() as u64;
}
