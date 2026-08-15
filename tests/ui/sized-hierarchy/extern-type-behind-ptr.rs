//@ check-pass
#![feature(extern_types, sized_hierarchy)]

use std::marker::{MetaSized, PointeeSized};

pub fn hash<T: PointeeSized>(_: *const T) {
    unimplemented!();
}

unsafe extern "C" {
    type Foo;
}

fn get() -> *const Foo {
    unimplemented!()
}

fn main() {
    hash::<Foo>(get());
}
