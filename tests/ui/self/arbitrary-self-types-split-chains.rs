//@ check-pass

#![feature(arbitrary_self_types)]
#![feature(arbitrary_self_types_split_chains)]

use std::ops::{Deref, Receiver};

struct A;
impl Deref for A {
    type Target = u8;
    fn deref(&self) -> &u8 {
        &0
    }
}
impl Receiver for A {
    type Target = u32;
}

struct B<'a, T>(&'a T);
impl<'a, T> Deref for B<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &*self.0
    }
}
impl<'a, T> Receiver for B<'a, T> {
    type Target = Box<T>;
}

fn main() {}
