//@ check-pass
//@ needs-rustc-debug-assertions

#![allow(unknown_or_malformed_diagnostic_attributes)]
#![allow(unused_macros)]

#[diagnostic::on_unmatched_args(message)]
macro_rules! pair {
    () => {};
}

fn main() {}
