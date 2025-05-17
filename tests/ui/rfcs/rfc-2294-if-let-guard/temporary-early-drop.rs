// issue-103476
//@ revisions: edition2021 edition2024
//@ [edition2021] edition: 2021
//@ [edition2024] edition: 2024
//@ check-pass

#![feature(if_let_guard)]
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
    match () {
        () if let Some(n) = m && let it = n.it() => {}
        _ => {}
    }
}

fn main() {}
