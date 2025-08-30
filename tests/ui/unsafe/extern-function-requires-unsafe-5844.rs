// https://github.com/rust-lang/rust/issues/5844
//@aux-build:aux-5844.rs

extern crate aux_5844;

fn main() {
    aux_5844::rand(); //~ ERROR: requires unsafe
}
