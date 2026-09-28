//@ build-pass
//@ aux-build:def_weak.rs

extern crate def_weak as dep;

fn main() {
    println!("{:p}", &dep::WEAK);
}
