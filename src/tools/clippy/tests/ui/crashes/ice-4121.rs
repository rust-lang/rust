//@ check-pass

use std::mem;

pub struct Foo<A, B>(A, B);

impl<A, B> Foo<A, B> {
    const HOST_SIZE: usize = mem::size_of::<B>();

    pub fn crash() -> bool {
        Self::HOST_SIZE == 0
    }
}

fn main() {}
