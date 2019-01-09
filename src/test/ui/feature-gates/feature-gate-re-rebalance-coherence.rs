// Test that the use of the box syntax is gated by `re-rebalance-coherence` feature gate.

// aux-build:re_rebalance_coherence_lib.rs

extern crate re_rebalance_coherence_lib as lib;
use lib::*;

struct Oracle;
impl Backend for Oracle {}
impl<'a, T:'a, Tab> QueryFragment<Oracle> for BatchInsert<'a, T, Tab> {}
//~^ ERROR E0210

fn main() {}
