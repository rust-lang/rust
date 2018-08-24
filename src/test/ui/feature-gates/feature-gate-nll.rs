#![feature(rustc_attrs)]
#![allow(dead_code)]

fn main() { #![rustc_error] // rust-lang/rust#49855
    let mut x = 33;

    let p = &x;
    x = 22; //~ ERROR cannot assign to `x` because it is borrowed [E0506]
}
