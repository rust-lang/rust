#![feature(plugin)]
#![plugin(clippy)]
#![deny(clippy, clippy_pedantic)]
#![allow(unused_imports, dead_code, missing_docs_in_private_items)]

use std::cmp::Ordering::*; //~ ERROR: don't use glob imports for enum variants

enum Enum {}

use self::Enum::*; //~ ERROR: don't use glob imports for enum variants

fn blarg() {
    use self::Enum::*; // ok, just for a function
}

mod blurg {
    pub use std::cmp::Ordering::*; // ok, re-export
}

mod tests {
    use super::*;
}

fn main() {}
