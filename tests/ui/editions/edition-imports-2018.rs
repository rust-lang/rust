// edition:2018
// aux-build:edition-imports-2015.rs

#[macro_use]
extern crate edition_imports_2015;

mod import {
    pub struct Path;
}
mod absolute {
    pub struct Path;
}

mod check {
    gen_imports!(); // OK

    fn check() {
        Path;
        LinkedList::<u8>::new();
    }
}

mod check_glob {
    gen_glob!(); //~ ERROR cannot glob-import all possible crates
}

fn main() {}
