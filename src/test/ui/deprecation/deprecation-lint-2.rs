// aux-build:deprecation-lint.rs
// error-pattern: use of deprecated item

#![deny(deprecated)]

#[macro_use]
extern crate deprecation_lint;

use deprecation_lint::*;

fn main() {
    macro_test!();
}
