#![feature(repr_simd)]

#[repr(simd)]
struct Bad([String; 2]); //~ ERROR E0077

fn main() {
}
