//! Use a gca const impl of a stable trait from a dependency as a pattern, without the feature.
//@ aux-build: gca-const-pattern-stable-trait.rs
//@ aux-build: gca-const-pattern-foreign-trait.rs
//@ run-pass

extern crate gca_const_pattern_foreign_trait as up;
extern crate gca_const_pattern_stable_trait as t;

fn main() {
    match 3usize {
        <up::S as t::Tr>::C => {}
        _ => panic!(),
    }
}
