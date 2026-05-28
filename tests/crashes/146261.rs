// This is part of series of regression tests for some diagnostics ICEs encountered in the wild with
// suggestions having overlapping parts under https://github.com/rust-lang/rust/pull/146121.

//@ needs-rustc-debug-assertions
//@ known-bug: #146261

enum U {
    B(),
}

fn main() {
    A(U::C)
}
