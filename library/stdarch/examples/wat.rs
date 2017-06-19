#![feature(target_feature)]

extern crate stdsimd;

use std::env;
use stdsimd::simd;

#[inline(never)]
#[target_feature = "-sse2"]
fn myop(
    (x0, x1, x2, x3): (u64, u64, u64, u64),
    (y0, y1, y2, y3): (u64, u64, u64, u64),
) -> (u64, u64, u64, u64) {
    let x = simd::u64x4::new(x0, x1, x2, x3);
    let y = simd::u64x4::new(y0, y1, y2, y3);
    let r = x * y;
    (r.extract(0), r.extract(1), r.extract(2), r.extract(3))
}

fn main() {
    let x = env::args().nth(1).unwrap().parse().unwrap();
    let y = env::args().nth(1).unwrap().parse().unwrap();
    let r = myop((x, x, x, x), (y, y, y, y));
    println!("{:?}", r);
}
