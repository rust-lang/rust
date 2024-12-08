//@ build-fail
#![feature(rustc_attrs)]

// Ensure that dyn-incompatible methods in Iterator does not generate
// vtable entries.

#[rustc_dump_vtable]
trait A: Iterator {}
//~^ error vtable

impl<T> A for T where T: Iterator {}

fn foo(_a: &mut dyn A<Item=u8>) {
}

fn main() {
    foo(&mut vec![0, 1, 2, 3].into_iter());
}
