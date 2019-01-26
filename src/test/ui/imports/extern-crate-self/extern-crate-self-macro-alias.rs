// run-pass

// Test that a macro can correctly expand the alias
// in an `extern crate self as ALIAS` item.

fn the_answer() -> usize { 42 }

macro_rules! alias_self {
    ($alias:ident) => { extern crate self as $alias; }
}

alias_self!(the_alias);

fn main() {
    assert_eq!(the_alias::the_answer(), 42);
}
