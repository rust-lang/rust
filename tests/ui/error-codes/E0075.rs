#![feature(repr_simd)]

#[repr(simd)]
struct Bad; //~ ERROR E0075

fn main() {
}
