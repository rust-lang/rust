//@ aux-build:unstable_feature.rs

extern crate unstable_feature; //~ ERROR: use of unstable library feature `feat_foo`
use unstable_feature::{Foo, Bar}; //~ ERROR: use of unstable library feature `feat_foo`
//~^ ERROR: use of unstable library feature `feat_foo`
//~^^ ERROR: use of unstable library feature `feat_foo`

fn main() { 
    Bar::foo(); //~ ERROR: type annotations needed: cannot satisfy `unstable feature: `feat_foo``
}