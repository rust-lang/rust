// build-pass

#![allow(incomplete_features)]
#![feature(adt_const_params)]

const F: f64 = 12.5;

pub fn func<const F: f64>() {}

fn main() {
    let _ = func::<14.0>();
    let _ = func::<F>();
}
