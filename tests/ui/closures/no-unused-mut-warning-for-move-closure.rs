//! Regression test for https://github.com/rust-lang/rust/issues/16671

//@ run-pass

#![deny(warnings)]

fn foo<F: FnOnce()>(_f: F) { }

fn main() {
    let mut var = Vec::new();
    foo(move|| {
        var.push(1);
    });
}
