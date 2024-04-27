//@ compile-flags:-Zpolymorphize=on
//@ build-pass

use std::any::TypeId;

pub fn foo<T: 'static>(_: T) -> TypeId {
    TypeId::of::<T>()
}

fn outer<T: 'static>() {
    foo(|| ());
}

fn main() {
    outer::<u8>();
}
