//@ edition: 2021

// Regression test for <https://github.com/rust-lang/rust/issues/117788>.
// Under some circumstances, the heuristics that detect macro name spans can
// get confused and produce incorrect spans beyond the bounds of the span
// being processed.

//@ aux-build: macro_name_span_helper.rs
extern crate macro_name_span_helper;

fn main() {
    affected_function();
}

macro_rules! macro_with_an_unreasonably_and_egregiously_long_name {
    () => {
        println!("hello");
    };
}

macro_name_span_helper::macro_that_defines_a_function! {
    fn affected_function() {
        macro_with_an_unreasonably_and_egregiously_long_name!();
    }
}
