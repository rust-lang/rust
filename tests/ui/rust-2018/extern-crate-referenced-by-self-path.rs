//@ run-pass
//@ aux-build:edition-lint-paths.rs
//@ run-rustfix

// Oddball: `edition_lint_paths` is accessed via this `self` path
// rather than being accessed directly. Unless we rewrite that path,
// we can't drop the extern crate.

#![deny(absolute_paths_not_starting_with_crate)]

extern crate edition_lint_paths;
use self::edition_lint_paths::foo;

fn main() {
    foo();
}
