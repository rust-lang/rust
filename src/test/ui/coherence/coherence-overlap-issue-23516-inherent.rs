// Tests that we consider `Box<U>: !Sugar` to be ambiguous, even
// though we see no impl of `Sugar` for `Box`. Therefore, an overlap
// error is reported for the following pair of impls (#23516).

// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]

pub trait Sugar {}

struct Cake<X>(X);

impl<T:Sugar> Cake<T> { fn dummy(&self) { } }
//[old]~^ ERROR E0592
//[re]~^^ ERROR E0592
impl<U:Sugar> Cake<Box<U>> { fn dummy(&self) { } }

fn main() { }
