#![feature(associated_consts)]

use std::marker::PhantomData;

trait Tr<'a> {
    const C: PhantomData<&'a u8> = PhantomData::<&'a u8>;
}

fn main() {}
