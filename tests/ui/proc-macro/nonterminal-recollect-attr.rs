//@ check-pass
//@ compile-flags: -Z span-debug
//@ proc-macro: nonterminal-recollect-attr.rs

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

extern crate nonterminal_recollect_attr;
use nonterminal_recollect_attr::*;

macro_rules! my_macro {
    ($v:ident) => {
        #[first_attr]
        $v struct Foo {
            field: u8
        }
    }
}

my_macro!(pub);
fn main() {}
