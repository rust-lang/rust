//! Regression test for https://github.com/rust-lang/rust/issues/34780

//@ check-pass

use std::marker::PhantomData;

trait Tr<'a> {
    const C: PhantomData<&'a u8> = PhantomData::<&'a u8>;
}

fn main() {}
