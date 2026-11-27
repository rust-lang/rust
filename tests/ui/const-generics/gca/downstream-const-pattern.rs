//! Use a gca const from a dependency as a pattern, without the feature.
//@ aux-build: gca-const-pattern-upstream.rs
//@ run-pass

extern crate gca_const_pattern_upstream as up;

fn main() {
    match 3usize {
        <up::S as up::Tr>::C => {}
        _ => panic!(),
    }
}
