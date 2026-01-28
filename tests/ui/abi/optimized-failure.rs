//! Unit parameter/return types should always be considered passed directly.
//!
//! Regression test for https://github.com/rust-lang/rust/issues/151748
//@ compile-flags: -O
//@ edition: 2018
//@ check-pass

fn main() {
    let _ = async || {};
    spawn(pattern_match());
}

fn spawn<F: Send>(_: F) {}

async fn pattern_match() {
    let COMPLEX_CONSTANT = ();
}

const fn do_nothing() {}

const COMPLEX_CONSTANT: () = do_nothing();
