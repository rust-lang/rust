//@ run-pass
//@ aux-build:macro_crate_nonterminal.rs

#[macro_use]
extern crate macro_crate_nonterminal as new_name;

pub fn main() {
    new_name::check_local();
    assert_eq!(increment!(5), 6);
}
