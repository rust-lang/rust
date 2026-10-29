//@ build-pass
//@ aux-build:def_external.rs

extern crate def_external as dep;

fn main() {
    println!("{:p}", &dep::EXTERN);
}
