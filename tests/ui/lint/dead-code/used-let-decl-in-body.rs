//@ check-pass

#![deny(dead_code)]

trait Tr {
    type X;
}

struct T;

impl Tr for T {
    type X = Self;
}

fn foo<T: Tr>() where {
    let _: T::X;
}

fn main() {
    foo::<T>();
}
