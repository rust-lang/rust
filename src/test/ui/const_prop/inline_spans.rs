// build-fail
// compile-flags: -Zmir-opt-level=3

#![deny(warnings)]

fn main() {
    let _ = add(u8::MAX, 1);
    //~^ NOTE in this expansion of inlined source
    //~| NOTE in this expansion of inlined source
}

#[inline(always)]
fn add(x: u8, y: u8) -> u8 {
    x + y
    //~^ ERROR this arithmetic operation will overflow
    //~| NOTE attempt to compute `u8::MAX + 1_u8`, which would overflow
    //~| NOTE `#[deny(arithmetic_overflow)]` on by default
}
