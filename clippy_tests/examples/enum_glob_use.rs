#![feature(plugin)]
#![plugin(clippy)]
#![warn(clippy, clippy_pedantic)]
#![allow(unused_imports, dead_code, missing_docs_in_private_items)]

use std::cmp::Ordering::*;

enum Enum {
    _Foo,
}

use self::Enum::*;

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
