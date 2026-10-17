//@ check-pass

#![feature(inherent_associated_types)]

struct D<T> {
  a: T
}

impl<T: Default> D<T> {
    type Item = T;

    fn next() -> Self::Item {
        Self::Item::default()
    }
}


fn main() {
}
