//@ check-pass
//@ revisions: allow expect

// this test checks that no matter if we put #[allow(dead_code)]
// or #[expect(dead_code)], no warning is being emitted

#![warn(dead_code)] // to override compiletest

fn f() {}

#[cfg_attr(allow, allow(dead_code))]
#[cfg_attr(expect, expect(dead_code))]
fn g() {
    f();
}

fn main() {}
