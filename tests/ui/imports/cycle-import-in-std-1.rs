//@ edition: 2018

// https://github.com/rust-lang/rust/issues/124490

use ops::{self as std};
//~^ ERROR: unresolved import `ops`
use std::collections::{self as ops};

fn main() {}
