// aux-build:edition-lint-paths.rs
// run-rustfix
// compile-flags:--extern edition_lint_paths
// edition:2018

// The "normal case". Ideally we would remove the `extern crate` here,
// but we don't.

#![deny(rust_2018_idioms)]
#![allow(dead_code)]

extern crate edition_lint_paths;
//~^ ERROR unused extern crate

extern crate edition_lint_paths as bar;
//~^ ERROR `extern crate` is not idiomatic in the new edition

fn main() {
    // This is not considered to *use* the `extern crate` in Rust 2018:
    use edition_lint_paths::foo;
    foo();

    // But this should be a use of the (renamed) crate:
    crate::bar::foo();
}

