//@ edition: 2021
//@ ignore-coverage-map
//@ ignore-windows
//@ llvm-cov-flags: --use-color

// Verify that telling `llvm-cov` to use colored output actually works.
// Ignored on Windows because we can't tell the tool to use ANSI escapes.

fn main() {
    for _i in 0..0 {}
}
