//@ revisions: old current
//@[old] edition: 2021
//@[current] edition: 2024
//@ aux-build: basic.rs
//@[current] check-pass

extern crate basic as edition_redirect;

mod downstream_ambiguity {
    use edition_redirect::ambiguity::alias_a::*;
    use edition_redirect::ambiguity::alias_b::*;

    fn check(_: Item) {}
    //[old]~^ ERROR `Item` is ambiguous
}

fn main() {}
