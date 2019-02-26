// edition:2015
// compile-flags:--extern absolute
// aux-build:edition-imports-2018.rs
// aux-build:absolute.rs

#[macro_use]
extern crate edition_imports_2018;

mod check {
    mod import {
        pub struct Path;
    }

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
