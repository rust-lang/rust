// https://github.com/rust-lang/rust/issues/56943
//@ aux-build:aux-56943.rs

extern crate aux_56943;

fn main() {
    let _: aux_56943::S = aux_56943::S2;
    //~^ ERROR mismatched types [E0308]
}
