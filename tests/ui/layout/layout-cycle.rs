//@ build-fail
//~^ ERROR: a cycle occurred during layout computation
//~| ERROR: cycle detected when computing layout of

// Issue #111176 -- ensure that we do not emit ICE on layout cycles

use std::mem;

pub struct S<T: Tr> {
    pub f: <T as Tr>::I,
}

pub trait Tr {
    type I: Tr;
}

impl<T: Tr> Tr for S<T> {
    type I = S<S<T>>;
}

impl Tr for () {
    type I = ();
}

fn foo<T: Tr>() -> usize {
    mem::size_of::<S<T>>()
}

fn main() {
    println!("{}", foo::<S<()>>());
}
