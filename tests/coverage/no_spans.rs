#![feature(coverage_attribute)]
//@ edition: 2021

// Test that coverage instrumentation can gracefully handle functions that end
// up having no relevant spans, without crashing the compiler or causing
// `llvm-cov` to fail.
//
// This was originally a regression test for issues such as #118643 and #118662.

fn main() {
    affected_function()();
}

macro_rules! macro_that_defines_a_function {
    (fn $name:ident () $body:tt) => {
        fn $name () -> impl Fn() $body
    }
}

macro_that_defines_a_function! {
    fn affected_function() {
        || ()
    }
}
