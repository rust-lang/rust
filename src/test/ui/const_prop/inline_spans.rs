// build-fail
// compile-flags: -Zmir-opt-level=2

#![deny(warnings)]

fn main() {
    let _ = add(u8::MAX, 1);
    //~^ ERROR this arithmetic operation will overflow
    let _ok = 1_u32.wrapping_sub(2_u32);
}

#[inline(always)]
fn add(x: u8, y: u8) -> u8 {
    x + y
}
