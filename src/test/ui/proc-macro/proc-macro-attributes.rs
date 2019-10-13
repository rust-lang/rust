// aux-build:derive-b.rs

#[macro_use]
extern crate derive_b;

#[B] //~ ERROR `B` is ambiguous
#[C] //~ ERROR cannot find attribute `C` in this scope
#[B(D)] //~ ERROR `B` is ambiguous
#[B(E = "foo")] //~ ERROR `B` is ambiguous
#[B(arbitrary tokens)] //~ ERROR `B` is ambiguous
#[derive(B)]
struct B;

fn main() {}
