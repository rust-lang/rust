#![feature(coverage_attribute)]
//@ edition: 2021

// If the span extractor can't find any relevant spans for a function, the
// refinement loop will terminate with nothing in its `prev` slot. If the
// subsequent code tries to unwrap `prev`, it will panic.
//
// This scenario became more likely after #118525 started discarding spans that
// can't be un-expanded back to within the function body.
//
// Regression test for "invalid attempt to unwrap a None some_prev", as seen
// in issues such as #118643 and #118662.

#[coverage(off)]
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
