//@ revisions: bfail1 bfail2

//@ compile-flags: -Znext-solver
//@ check-pass

#![allow(dead_code)]

pub trait Future {
    type Error;
    fn poll() -> Self::Error;
}

struct S;
impl Future for S {
    type Error = Error;
    fn poll() -> Self::Error {
        todo!()
    }
}

#[cfg(bfail1)]
pub struct Error(());

#[cfg(bfail2)]
pub struct Error();

fn main() {}
