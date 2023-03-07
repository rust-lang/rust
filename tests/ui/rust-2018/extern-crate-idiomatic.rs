// run-pass
// aux-build:edition-lint-paths.rs
// compile-flags:--extern edition_lint_paths
// run-rustfix

// The "normal case". Ideally we would remove the `extern crate` here,
// but we don't.

#![feature(rust_2018_preview)]
#![deny(absolute_paths_not_starting_with_crate)]

extern crate edition_lint_paths;

use edition_lint_paths::foo;

fn main() {
    foo();
}
