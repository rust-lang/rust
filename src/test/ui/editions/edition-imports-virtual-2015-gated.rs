// edition:2018
// aux-build:edition-imports-2015.rs

#[macro_use]
extern crate edition_imports_2015;

mod check {
    gen_gated!(); //~ ERROR unresolved import `E`
}

fn main() {}
