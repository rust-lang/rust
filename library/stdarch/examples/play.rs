extern crate stdsimd;

use std::env;
use std::io::Write;

use stdsimd as s;

fn main() {
    let arg1: f64 = env::args().nth(1).unwrap().parse().unwrap();
    let arg2: f64 = env::args().nth(2).unwrap().parse().unwrap();
    let arg3: f64 = env::args().nth(3).unwrap().parse().unwrap();
    let arg4: f64 = env::args().nth(4).unwrap().parse().unwrap();
    unsafe {
        let a1 = s::_mm_load_pd(&(arg1, arg2) as *const _ as *const f64);
        let b1 = s::_mm_load_pd(&(arg3, arg4) as *const _ as *const f64);
        // println!("{:?}, {:?}", a, b);
        let r1 = s::_mm_add_sd(a1, b1);
        // println!("{:?}", r1);
        let mut r2: (f64, f64) = (0.0, 0.0);
        s::_mm_store_pd(&mut r2 as *mut _ as *mut f64, r1);
        if r2 == (4.0, 2.0) {
            ::std::io::stdout().write_all(b"yes\n").unwrap();
        } else {
            ::std::io::stdout().write_all(b"NO\n").unwrap();
        }
        // println!("{:?}", r2);
    }
}
