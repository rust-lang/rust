//@ run-pass
//@ compile-flags: -Zmir-opt-level=5 -Zvalidate-mir

#![feature(ptr_metadata)]

// Regression for <https://github.com/rust-lang/rust/issues/125506>,
// which failed because of SRoA would project into `DynMetadata`.

trait Foo {}

struct Bar;

impl Foo for Bar {}

fn main() {
    let a: *mut dyn Foo = &mut Bar;

    let _d = a.to_raw_parts().0 as usize;
}
