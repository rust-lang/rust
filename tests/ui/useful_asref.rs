#![deny(clippy::useless_asref)]

trait Trait {
    fn as_ptr(&self);
}

impl<'a> Trait for &'a [u8] {
    fn as_ptr(&self) {
        self.as_ref().as_ptr();
    }
}

fn main() {}
