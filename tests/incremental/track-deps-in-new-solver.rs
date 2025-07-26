//@ revisions: cfail1 cfail2

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

#[cfg(cfail1)]
pub struct Error(());

#[cfg(cfail2)]
pub struct Error();

fn main() {}
