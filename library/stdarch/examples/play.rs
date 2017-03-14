#![feature(target_feature)]

extern crate stdsimd;

use std::env;
use stdsimd as s;

#[inline(never)]
#[target_feature = "+sse4.2"]
fn index(needle: &str, haystack: &str) -> usize {
    assert!(needle.len() <= 16 && haystack.len() <= 16);

    let (needle_len, hay_len) = (needle.len(), haystack.len());

    let mut needle = needle.to_string().into_bytes();
    needle.resize(16, 0);
    let vneedle = s::__m128i::from(s::u8x16::load(&needle, 0));

    let mut haystack = haystack.to_string().into_bytes();
    haystack.resize(16, 0);
    let vhaystack = s::__m128i::from(s::u8x16::load(&haystack, 0));

    s::_mm_cmpestri(
        vneedle, needle_len as i32, vhaystack, hay_len as i32,
        s::_SIDD_CMP_EQUAL_ORDERED) as usize
}

fn main() {
    // let x0: f64 = env::args().nth(1).unwrap().parse().unwrap();
    // let x1: f64 = env::args().nth(2).unwrap().parse().unwrap();
    // let x2: f64 = env::args().nth(3).unwrap().parse().unwrap();
    // let x3: f64 = env::args().nth(4).unwrap().parse().unwrap();
    // let y0: i32 = env::args().nth(5).unwrap().parse().unwrap();
    // let y1: i32 = env::args().nth(6).unwrap().parse().unwrap();
    // let y2: i32 = env::args().nth(7).unwrap().parse().unwrap();
    // let y3: i32 = env::args().nth(8).unwrap().parse().unwrap();

    // let a = s::f64x2::new(x0, x1);
    // let b = s::f64x2::new(x2, x3);
    // let r = s::_mm_cmplt_sd(a, b);
    // let r = foobar(a, b);


    let needle = env::args().nth(1).unwrap();
    let haystack = env::args().nth(2).unwrap();
    println!("{:?}", index(&needle, &haystack));
}
