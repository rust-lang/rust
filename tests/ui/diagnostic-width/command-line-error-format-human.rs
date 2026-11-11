//! This test checks the output format without the intermediate json representation

//@ compile-flags: --error-format=human

pub fn main() {
    let x = 42;
    x = 43;
}
