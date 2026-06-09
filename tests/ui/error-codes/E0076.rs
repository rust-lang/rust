#![feature(repr_simd)]

#[repr(simd)]
struct Bad(u32);
//~^ ERROR E0076

fn main() {
}
