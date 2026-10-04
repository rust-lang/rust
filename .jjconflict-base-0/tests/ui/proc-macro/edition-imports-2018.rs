//@ check-pass
//@ edition:2018
//@ proc-macro: edition-imports-2015.rs
//@ ignore-backends: gcc

#[macro_use]
extern crate edition_imports_2015;

mod import {
    pub struct Path;
}
mod absolute {
    pub struct Path;
}

mod check {
    #[derive(Derive2015)] // OK
    struct S;

    fn check() {
        Path;
    }
}

fn main() {}
