//! Regression test for <https://github.com/rust-lang/rust/issues/98077>.

//@ edition:2018
//@ check-pass

#![allow(dead_code)]
#![allow(unused_assignments)]

async fn foo() {
    let mut f = None;
    let value = 0;
    f = Some(async { value });
    core::mem::drop(f);
}

fn main() { }
