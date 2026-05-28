//@ edition:2015
use not_existing_crate::*; //~ ERROR unresolved import `not_existing_crate
use std as foo;

fn main() {}
