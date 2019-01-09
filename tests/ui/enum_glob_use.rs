#![warn(clippy::all, clippy::pedantic)]
#![allow(unused_imports, dead_code, clippy::missing_docs_in_private_items)]

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

#[allow(non_snake_case)]
mod CamelCaseName {}

use CamelCaseName::*;

fn main() {}
