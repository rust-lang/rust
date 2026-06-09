#![allow(deprecated)]

use core::hash::*;

use test::{Bencher, black_box};

fn hash_bytes<H: Hasher>(mut s: H, x: &[u8]) -> u64 {
    Hasher::write(&mut s, x);
    s.finish()
}

fn hash_with<H: Hasher, T: Hash>(mut st: H, x: &T) -> u64 {
    x.hash(&mut st);
    st.finish()
}

fn hash<T: Hash>(x: &T) -> u64 {
    hash_with(SipHasher::new(), x)
}

#[bench]
fn bench_str_under_8_bytes(b: &mut Bencher) {
    let s = "foo";
    b.iter(|| {
        assert_eq!(hash(&s), 16262950014981195938);
    })
}

#[bench]
fn bench_str_of_8_bytes(b: &mut Bencher) {
    let s = "foobar78";
    b.iter(|| {
        assert_eq!(hash(&s), 4898293253460910787);
    })
}

#[bench]
fn bench_str_over_8_bytes(b: &mut Bencher) {
    let s = "foobarbaz0";
    b.iter(|| {
        assert_eq!(hash(&s), 10581415515220175264);
    })
}

#[bench]
fn bench_long_str(b: &mut Bencher) {
    let s = "Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor \
             incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud \
             exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute \
             irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla \
             pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui \
             officia deserunt mollit anim id est laborum.";
    b.iter(|| {
        assert_eq!(hash(&s), 17717065544121360093);
    })
}

#[bench]
fn bench_u32(b: &mut Bencher) {
    let u = 162629500u32;
    let u = black_box(u);
    b.iter(|| hash(&u));
    b.bytes = 8;
}

#[bench]
fn bench_u32_keyed(b: &mut Bencher) {
    let u = 162629500u32;
    let u = black_box(u);
    let k1 = black_box(0x1);
    let k2 = black_box(0x2);
    b.iter(|| hash_with(SipHasher::new_with_keys(k1, k2), &u));
    b.bytes = 8;
}

#[bench]
fn bench_u64(b: &mut Bencher) {
    let u = 16262950014981195938u64;
    let u = black_box(u);
    b.iter(|| hash(&u));
    b.bytes = 8;
}

#[bench]
fn bench_bytes_4(b: &mut Bencher) {
    let data = black_box([b' '; 4]);
    b.iter(|| hash_bytes(SipHasher::default(), &data));
    b.bytes = 4;
}

#[bench]
fn bench_bytes_7(b: &mut Bencher) {
    let data = black_box([b' '; 7]);
    b.iter(|| hash_bytes(SipHasher::default(), &data));
    b.bytes = 7;
}

#[bench]
fn bench_bytes_8(b: &mut Bencher) {
    let data = black_box([b' '; 8]);
    b.iter(|| hash_bytes(SipHasher::default(), &data));
    b.bytes = 8;
}

#[bench]
fn bench_bytes_a_16(b: &mut Bencher) {
    let data = black_box([b' '; 16]);
    b.iter(|| hash_bytes(SipHasher::default(), &data));
    b.bytes = 16;
}

#[bench]
fn bench_bytes_b_32(b: &mut Bencher) {
    let data = black_box([b' '; 32]);
    b.iter(|| hash_bytes(SipHasher::default(), &data));
    b.bytes = 32;
}

#[bench]
fn bench_bytes_c_128(b: &mut Bencher) {
    let data = black_box([b' '; 128]);
    b.iter(|| hash_bytes(SipHasher::default(), &data));
    b.bytes = 128;
}
