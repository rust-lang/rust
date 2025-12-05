#![cfg_attr(feat, feature(coverage_attribute))]
//@ revisions: feat nofeat
//@ compile-flags: -Cinstrument-coverage
//@ needs-profiler-runtime
//@ reference: attributes.coverage.syntax

// Malformed `#[coverage(..)]` attributes should not cause an ICE when built
// with `-Cinstrument-coverage`.
// Regression test for <https://github.com/rust-lang/rust/issues/127880>.

#[coverage]
//~^ ERROR malformed `coverage` attribute input
//[nofeat]~| ERROR the `#[coverage]` attribute is an experimental feature
fn main() {}

// FIXME(#130766): When the `#[coverage(..)]` attribute is stabilized,
// get rid of the revisions and just make this a normal test.
