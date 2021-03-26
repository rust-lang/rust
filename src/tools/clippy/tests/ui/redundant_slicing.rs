#![allow(unused)]
#![warn(clippy::redundant_slicing)]

fn main() {
    let x: &[u32] = &[0];
    let err = &x[..];

    let v = vec![0];
    let ok = &v[..];
    let err = &(&v[..])[..];
}
