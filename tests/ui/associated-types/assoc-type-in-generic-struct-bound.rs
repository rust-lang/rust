//! Regression test for https://github.com/rust-lang/rust/issues/19632

//@ check-pass
#![allow(dead_code)]

trait PoolManager {
    type C;
    fn dummy(&self) { }
}

struct InnerPool<M: PoolManager> {
    manager: M,
}

fn main() {}
