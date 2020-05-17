// aux-build:test-macros.rs

extern crate test_macros;

#[test_macros::recollect_attr]
fn repro() {
    f :: < Vec < _ > > ( ) ; //~ ERROR cannot find
}
fn main() {}
