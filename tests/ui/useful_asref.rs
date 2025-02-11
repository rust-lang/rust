//@ check-pass

#![deny(clippy::useless_asref)]
#![allow(clippy::needless_lifetimes)]

trait Trait {
    fn as_ptr(&self);
}

impl<'a> Trait for &'a [u8] {
    fn as_ptr(&self) {
        self.as_ref().as_ptr();
    }
}

fn main() {}
