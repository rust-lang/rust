// Test that a local type (with no type parameters) appearing within a
// *non-fundamental* remote type like `Vec` is not considered local.

//@ aux-build:coherence_lib.rs

extern crate coherence_lib as lib;
use lib::Remote;

struct Local;

impl Remote for Vec<Local> { }
//~^ ERROR E0117

fn main() { }
