//@check-pass
#![warn(clippy::mutable_key_type)]

use std::marker::PhantomData;

trait Group {
    type ExposantSet: Group;
}

struct Pow<T: Group> {
    exposant: Box<Pow<T::ExposantSet>>,
    _p: PhantomData<T>,
}

impl<T: Group> Pow<T> {
    fn is_zero(&self) -> bool {
        false
    }
    fn normalize(&self) {
        #[expect(clippy::if_same_then_else)]
        if self.is_zero() {
        } else if false {
        }
    }
}

fn main() {}
