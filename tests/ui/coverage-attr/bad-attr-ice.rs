//@ compile-flags: -Cinstrument-coverage
//@ needs-profiler-runtime

// Malformed `#[coverage(..)]` attributes should not cause an ICE when built
// with `-Cinstrument-coverage`.
// Regression test for <https://github.com/rust-lang/rust/issues/127880>.

#[coverage]
//~^ ERROR malformed `coverage` attribute input
fn main() {}
