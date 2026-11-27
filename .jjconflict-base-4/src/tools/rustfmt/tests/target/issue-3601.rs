#![feature(const_generics)]

trait A {
    fn foo(&self);
}

pub struct B<const N: usize>([usize; N]);

impl<const N: usize> A for B<{ N }> {
    fn foo(&self) {}
}
