// issue-103476
//@ compile-flags: -Zlint-mir
//@ edition: 2024
//@ check-pass

#![allow(irrefutable_let_patterns)]

struct Pd;

impl Pd {
    fn it(&self) -> It {
        todo!()
    }
}

pub struct It<'a>(Box<dyn Tr<'a>>);

trait Tr<'a> {}

fn f(m: Option<Pd>) {
    if let Some(n) = m && let it = n.it() {};
}

fn main() {}
