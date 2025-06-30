//@ run-rustfix
//@ edition: 2018

#![deny(unused_parens)]

pub type DynIsAContextualKeywordIn2015 = Box<dyn (::std::ops::Fn())>; //~ ERROR unnecessary parentheses around type

fn main() {}
