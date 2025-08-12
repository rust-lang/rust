// https://github.com/rust-lang/rust/issues/87707
// test for #87707
//@ edition:2018
//@ run-fail
//@ exec-env:RUST_BACKTRACE=0
//@ check-run-results
//@ needs-unwind uses catch_unwind

use std::sync::Once;
use std::panic;

fn main() {
    let o = Once::new();
    let _ = panic::catch_unwind(|| {
        o.call_once(|| panic!("Here Once instance is poisoned."));
    });
    o.call_once(|| {});
}
