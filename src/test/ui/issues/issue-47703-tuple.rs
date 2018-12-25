// compile-pass
#![allow(dead_code)]
#![feature(nll)]

struct WithDrop;

impl Drop for WithDrop {
    fn drop(&mut self) {}
}

fn consume(x: (&mut (), WithDrop)) -> &mut () { x.0 }

fn main() {}
