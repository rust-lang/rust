// edition:2018

#![deny(unused)]

use std::fmt;

// No "unresolved import" + "unused import" combination here.
use fmt::Write; //~ ERROR imports can only refer to extern crate names
                //~| ERROR unused import: `fmt::Write`

fn main() {}
