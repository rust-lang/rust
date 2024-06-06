//@ edition: 2018

// https://github.com/rust-lang/rust/issues/124490

use io::{self as std};
//~^ ERROR: unresolved import `io`
use std::collections::{self as io};

fn main() {}
