//@ check-pass

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![warn(clippy::branches_sharing_code)]

const fn f() -> usize {
    2
}
const C: [f64; f()] = [0f64; f()];

fn main() {
    let _ = if true { C[0] } else { C[1] };
}
