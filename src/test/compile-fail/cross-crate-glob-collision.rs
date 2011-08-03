// error-pattern:'swap' is glob-imported from multiple different modules
// issue #482

use std;
// expecting swap to be defined in vec
import std::vec::*;
import alternate_supplier::*;

mod alternate_supplier {
    fn swap() { }
}

fn main() { swap() }