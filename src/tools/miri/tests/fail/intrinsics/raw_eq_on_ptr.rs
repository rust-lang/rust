#![feature(intrinsics)]

extern "rust-intrinsic" {
    fn raw_eq<T>(a: &T, b: &T) -> bool;
}

fn main() {
    let x = &0;
    unsafe { raw_eq(&x, &x) }; //~ERROR: `raw_eq` on bytes with provenance
}
