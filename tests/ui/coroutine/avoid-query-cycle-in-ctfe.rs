//! Items whose type depends on CTFE (such as the async closure/coroutine beneath, whose type
//! depends upon evaluating `do_nothing`) should not cause a query cycle owing to the deduction of
//! the function's parameter attributes, which are only required for codegen and not for CTFE.
//!
//! Regression test for https://github.com/rust-lang/rust/issues/151748
//@ compile-flags: -O
//@ edition: 2018
//@ check-pass

fn main() {
    let _ = async || {
        let COMPLEX_CONSTANT = ();
    };
}

const fn do_nothing() {}

const COMPLEX_CONSTANT: () = do_nothing();
