//@ aux-build:coherence_lib.rs

extern crate coherence_lib as lib;
use lib::{Remote, Pair};

struct Local<T>(T);

impl<T,U> Remote for Pair<T,Local<U>> { }
//~^ ERROR E0117

fn main() { }
