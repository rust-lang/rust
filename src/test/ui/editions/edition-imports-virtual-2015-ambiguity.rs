// build-pass (FIXME(62277): could be check-pass?)
// edition:2018
// compile-flags:--extern edition_imports_2015
// aux-build:edition-imports-2015.rs

mod edition_imports_2015 {
    pub struct Path;
}

pub struct Ambiguous {}

mod check {
    pub struct Ambiguous {}

    fn check() {
        edition_imports_2015::gen_ambiguous!(); // OK
    }
}

fn main() {}
