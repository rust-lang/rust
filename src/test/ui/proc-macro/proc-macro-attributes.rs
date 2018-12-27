// aux-build:derive-b.rs

#[macro_use]
extern crate derive_b;

#[B]
#[C] //~ ERROR attribute `C` is currently unknown to the compiler
#[B(D)]
#[B(E = "foo")]
#[B(arbitrary tokens)]
#[derive(B)]
struct B;

fn main() {}
