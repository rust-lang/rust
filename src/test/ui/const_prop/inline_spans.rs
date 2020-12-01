// build-fail
// compile-flags: -Zmir-opt-level=2

#![deny(warnings)]

fn main() {
    let _ = add(u8::MAX, 1);
    //~^ ERROR this arithmetic operation will overflow
}

#[inline(always)]
fn add(x: u8, y: u8) -> u8 {
    x + y
}
