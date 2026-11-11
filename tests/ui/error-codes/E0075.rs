#![feature(repr_simd)]

#[repr(simd)]
struct Bad; //~ ERROR E0075

#[repr(simd)]
struct AlsoBad([i32; 1], [i32; 1]); //~ ERROR E0075

fn main() {
}
