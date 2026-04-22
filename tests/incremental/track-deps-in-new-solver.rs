//@ revisions: cpass1 cpass2
//@ compile-flags: -Znext-solver

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

#[cfg(cpass1)]
pub struct Error(());

#[cfg(cpass2)]
pub struct Error();

fn main() {}
