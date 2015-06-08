#![feature(test)]

extern crate test;
extern crate rustc_serialize;

use rustc_serialize::base64::{FromBase64, ToBase64, STANDARD};
use test::Bencher;

#[bench]
fn bench_to_base64(b: &mut Bencher) {
    let s = "イロハニホヘト チリヌルヲ ワカヨタレソ ツネナラム \
             ウヰノオクヤマ ケフコエテ アサキユメミシ ヱヒモセスン";
    b.iter(|| {
        s.as_bytes().to_base64(STANDARD);
    });
    b.bytes = s.len() as u64;
}

#[bench]
fn bench_from_base64(b: &mut Bencher) {
    let s = "イロハニホヘト チリヌルヲ ワカヨタレソ ツネナラム \
             ウヰノオクヤマ ケフコエテ アサキユメミシ ヱヒモセスン";
    let sb = s.as_bytes().to_base64(STANDARD);
    b.iter(|| {
        sb.from_base64().unwrap();
    });
    b.bytes = sb.len() as u64;
}

