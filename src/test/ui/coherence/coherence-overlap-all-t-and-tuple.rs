// Check that we detect an overlap here in the case where:
//
//    for some type X:
//      T = (X,)
//      T11 = X, U11 = X
//
// Seems pretty basic, but then there was issue #24241. :)

// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]

trait From<U> {
    fn foo() {}
}

impl <T> From<T> for T {
}

impl <T11, U11> From<(U11,)> for (T11,) {
//[old]~^ ERROR E0119
//[re]~^^ ERROR E0119
}

fn main() { }
