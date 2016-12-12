extern crate stdsimd;

use std::env;
use stdsimd as s;

#[inline(never)]
fn foobar(a: s::f64x2, b: s::f64x2) -> s::f64x2 {
    s::_mm_cmpge_sd(a, b)
}

fn main() {
    let x0: f64 = env::args().nth(1).unwrap().parse().unwrap();
    let x1: f64 = env::args().nth(2).unwrap().parse().unwrap();
    let x2: f64 = env::args().nth(3).unwrap().parse().unwrap();
    let x3: f64 = env::args().nth(4).unwrap().parse().unwrap();
    // let y0: i32 = env::args().nth(5).unwrap().parse().unwrap();
    // let y1: i32 = env::args().nth(6).unwrap().parse().unwrap();
    // let y2: i32 = env::args().nth(7).unwrap().parse().unwrap();
    // let y3: i32 = env::args().nth(8).unwrap().parse().unwrap();

    let a = s::f64x2::new(x0, x1);
    let b = s::f64x2::new(x2, x3);
    // let r = s::_mm_cmplt_sd(a, b);
    let r = foobar(a, b);
    println!("{:?}", r);
}
