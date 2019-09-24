// build-pass (FIXME(62277): could be check-pass?)
#![allow(stable_features)]
#![feature(associated_consts)]

use std::marker::PhantomData;

trait Tr<'a> {
    const C: PhantomData<&'a u8> = PhantomData::<&'a u8>;
}

fn main() {}
