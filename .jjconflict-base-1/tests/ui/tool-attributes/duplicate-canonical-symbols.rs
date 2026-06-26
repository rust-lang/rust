//@ aux-build: symbols.rs

#![feature(rustc_attrs)]

extern crate symbols;

extern "C" {
    #[rustc_canonical_symbol]
    fn foo();
}

mod bar {
    extern "C" {
        #[rustc_canonical_symbol]
        fn foo();
        //~^ ERROR duplicate canonical symbol
        //~^^ ERROR duplicate canonical symbol
    }
}

fn main() {}
