#![feature(rustc_attrs)]

// Ensure that dyn-incompatible methods in Iterator does not generate
// vtable entries.

trait A: Iterator {}

impl<T> A for T where T: Iterator {}

#[rustc_dump_vtable]
type Test = dyn A<Item=u8>;
//~^ error vtable

fn foo(_a: &mut dyn A<Item=u8>) {
}

fn main() {
    foo(&mut vec![0, 1, 2, 3].into_iter());
}
