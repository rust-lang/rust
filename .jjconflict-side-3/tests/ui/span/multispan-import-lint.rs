//@ check-pass

#![warn(unused)]

use std::cmp::{Eq, Ord, min, PartialEq, PartialOrd};
//~^ WARN unused imports

fn main() {
    let _ = min(1, 2);
}
