//@ check-pass

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

fn user<T: Copy>() where S<T>::P: std::fmt::Debug {}

struct S<T>(T);

impl<T: Copy> S<T> {
    type P = ();
}

fn main() {}
