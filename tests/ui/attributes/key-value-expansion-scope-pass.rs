// Imports suppress the `out_of_scope_macro_calls` lint.

//@ check-pass
//@ edition:2018

#![doc = in_root!()]

macro_rules! in_root { () => { "" } }
use in_root;

mod macros_stay {
    #![doc = in_mod!()]

    macro_rules! in_mod { () => { "" } }
    use in_mod;
}

fn main() {}
