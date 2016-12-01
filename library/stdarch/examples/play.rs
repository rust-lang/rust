/*
extern crate stdsimd;

use std::env;

use stdsimd as s;

fn main() {
    let arg1: u8 = env::args().nth(1).unwrap().parse().unwrap();
    let arg2: u8 = env::args().nth(2).unwrap().parse().unwrap();
    let arg3: u8 = env::args().nth(3).unwrap().parse().unwrap();
    // let arg4: u8 = env::args().nth(4).unwrap().parse().unwrap();
    unsafe {
        s::_mm_lfence();
        s::_mm_pause();
        let a = s::u8x16::new(
            arg1, arg1, arg1, arg1, arg1, arg1, arg1, arg1,
            arg2, arg2, arg2, arg2, arg2, arg2, arg2, arg2);
        // let b = s::u8x16::new(
            // arg3, arg3, arg3, arg3, arg3, arg3, arg3, arg3,
            // arg4, arg4, arg4, arg4, arg4, arg4, arg4, arg4);
        let r = s::_mm_srli_si128(a.as_m128i(), arg3 as i32);
        println!("{:?}", s::u8x16::from(r));
    }
}
*/
fn main(){}
