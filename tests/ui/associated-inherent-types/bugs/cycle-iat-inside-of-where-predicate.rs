// known-bug: unknown

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

// FIXME(inherent_associated_types): This shouldn't lead to a cycle error.

fn user<T>() where S<T>::P: std::fmt::Debug {}

struct S<T>;

impl<T: Copy> S<T> {
    type P = ();
}

fn main() {}
