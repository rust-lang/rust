#![feature(repr_simd)]

#[repr(simd)]
struct Bad(String); //~ ERROR E0077

fn main() {
}
