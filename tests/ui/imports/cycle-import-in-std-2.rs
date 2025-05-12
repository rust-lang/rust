//@ edition: 2018

// https://github.com/rust-lang/rust/issues/125013

use ops::{self as std};
//~^ ERROR: unresolved import `ops`
use std::ops::Deref::{self as ops};

fn main() {}
