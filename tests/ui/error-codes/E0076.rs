#![feature(repr_simd)]

#[repr(simd)]
struct Bad(u16, u32, u32);
//~^ ERROR E0076

fn main() {
}
