// Must be build-pass, because check-pass will not run const prop and thus not emit the lint anyway.
//@ build-pass
//@ compile-flags: -Zmir-opt-level=3

#![deny(warnings)]

fn main() {
    #[allow(arithmetic_overflow)]
    let _ = add(u8::MAX, 1);
}

#[inline(always)]
fn add(x: u8, y: u8) -> u8 {
    x + y
}
