// edition:2018
// aux-build:edition-imports-2015.rs
// error-pattern: `Ambiguous` is ambiguous

#[macro_use]
extern crate edition_imports_2015;

pub struct Ambiguous {}

mod check {
    pub struct Ambiguous {}

    fn check() {
        gen_ambiguous!();
    }
}

fn main() {}
