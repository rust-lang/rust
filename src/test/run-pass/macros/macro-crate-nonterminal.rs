// run-pass
// aux-build:macro_crate_nonterminal.rs

#[macro_use]
extern crate macro_crate_nonterminal;

pub fn main() {
    macro_crate_nonterminal::check_local();
    assert_eq!(increment!(5), 6);
}
