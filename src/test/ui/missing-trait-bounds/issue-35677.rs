// run-rustfix
#![allow(dead_code)]
use std::collections::HashSet;
use std::hash::Hash;

fn is_subset<T>(this: &HashSet<T>, other: &HashSet<T>) -> bool {
    this.is_subset(other)
    //~^ ERROR the method
}

fn main() {}
