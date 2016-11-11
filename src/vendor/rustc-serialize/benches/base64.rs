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


#[bench]
fn bench_to_base64_large(b: &mut Bencher) {
    let s: Vec<_> = (0..10000).map(|i| ((i as u32 * 12345) % 256) as u8).collect();
    b.iter(|| {
        s.to_base64(STANDARD);
    });
    b.bytes = s.len() as u64;
}

#[bench]
fn bench_from_base64_large(b: &mut Bencher) {
    let s: Vec<_> = (0..10000).map(|i| ((i as u32 * 12345) % 256) as u8).collect();
    let sb = s.to_base64(STANDARD);
    b.iter(|| {
        sb.from_base64().unwrap();
    });
    b.bytes = sb.len() as u64;
}
