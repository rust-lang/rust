// Regression test of #43913.

//@ run-rustfix

#![feature(trait_alias)]
#![allow(bare_trait_objects, dead_code)]

type Strings = Iterator<Item=String>;

struct Struct<S: Strings>(S);
//~^ ERROR: expected trait, found type alias `Strings`

fn main() {}
