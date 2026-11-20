//@ edition: 2021

// If the span extractor can't find any relevant spans for a function,
// but the function contains coverage span-marker statements (e.g. inserted
// for `if !`), coverage codegen may think that it is instrumented and
// consequently complain that it has no spans.
//
// Regression test for <https://github.com/rust-lang/rust/issues/118850>,
// "A used function should have had coverage mapping data but did not".

fn main() {
    affected_function();
}

macro_rules! macro_that_defines_a_function {
    (fn $name:ident () $body:tt) => {
        fn $name () $body
    }
}

macro_that_defines_a_function! {
    fn affected_function() {
        if !false {
            ()
        } else {
            ()
        }
    }
}
