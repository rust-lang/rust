//@ proc-macro: test-macros.rs
//@ check-pass

#[macro_use]
extern crate test_macros;

#[derive(Print)]
enum ProceduralMasqueradeDummyType {
    Input
}

fn main() {}
