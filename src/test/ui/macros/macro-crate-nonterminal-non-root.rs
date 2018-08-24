// aux-build:macro_crate_nonterminal.rs

mod foo {
    #[macro_use]
    extern crate macro_crate_nonterminal;  //~ ERROR must be at the crate root
}

fn main() {
}
