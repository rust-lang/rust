// edition:2018
// aux-build:edition-imports-2015.rs
// error-pattern: imports can only refer to extern crate names passed with `--extern`

#[macro_use]
extern crate edition_imports_2015;

mod check {
    gen_gated!();
}

fn main() {}
