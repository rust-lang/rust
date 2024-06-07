//@ edition: 2018

// https://github.com/rust-lang/rust/issues/125013

use io::{self as std};
//~^ ERROR: unresolved import `io`
use std::ops::Deref::{self as io};

fn main() {}
