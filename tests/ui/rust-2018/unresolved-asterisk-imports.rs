// edition:2018

use nonexistent_crate::*; //~ ERROR unresolved import `nonexistent_crate
use std as foo;

fn main() {}
