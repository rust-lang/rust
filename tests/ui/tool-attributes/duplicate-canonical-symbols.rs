//@ aux-build: symbols.rs

#![feature(rustc_attrs)]

extern crate symbols;

extern "C" {
    #[rustc_canonical_symbol = "foo"]
    fn foo();

    #[rustc_canonical_symbol = "foo"]
    fn foo2();
    //~^ ERROR duplicate canonical symbol
    //~^^ ERROR duplicate canonical symbol
}

fn main() {}
