// rust-lang/rust#59548: We used to ICE when trying to use a static
// with a type that violated its own `#[linkage]`.

// aux-build:def_illtyped_external.rs

extern crate def_illtyped_external as dep;

fn main() {
    println!("{:p}", &dep::EXTERN);
}
