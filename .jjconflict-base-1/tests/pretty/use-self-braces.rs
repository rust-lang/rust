//@ pp-exact
//@ edition:2021

#![allow(unused_imports)]

// Braces around `self` must be preserved, because `use foo::self` is not valid Rust.
use std::io::{self};
use std::fmt::{self, Debug};

fn main() {}
