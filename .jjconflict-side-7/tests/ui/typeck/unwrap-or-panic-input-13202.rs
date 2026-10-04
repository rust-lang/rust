//! Regression test for https://github.com/rust-lang/rust/issues/13202

//@ run-fail
//@ error-pattern:bad input
//@ needs-subprocess

fn main() {
    Some("foo").unwrap_or(panic!("bad input")).to_string();
}
