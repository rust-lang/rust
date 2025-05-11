//@ compile-flags:-Zmir-opt-level=0

#![deny(dead_code)]
#![feature(coerce_unsized)]
#![feature(unsize)]
#![crate_type = "lib"]

use std::marker::Unsize;
use std::ops::CoerceUnsized;

trait Trait {
    fn foo(&self);
}

// Simple Case
impl Trait for bool {
    fn foo(&self) {}
}

impl Trait for char {
    fn foo(&self) {}
}

// Struct Field Case
struct Struct<T: ?Sized> {
    _a: u32,
    _b: i32,
    _c: T,
}

impl Trait for f64 {
    fn foo(&self) {}
}

// Custom Coercion Case
impl Trait for u32 {
    fn foo(&self) {}
}

#[derive(Clone, Copy)]
struct Wrapper<T: ?Sized>(#[allow(dead_code)] *const T);

impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<Wrapper<U>> for Wrapper<T> {}

//~ MONO_ITEM fn start
#[no_mangle]
pub fn start(_: isize, _: *const *const u8) -> isize {
    // simple case
    let bool_sized = &true;
    //~ MONO_ITEM fn std::ptr::drop_in_place::<bool> - shim(None) @@ unsizing-cgu.0[Internal]
    //~ MONO_ITEM fn <bool as Trait>::foo
    let _bool_unsized = bool_sized as &Trait;

    let char_sized = &'a';

    //~ MONO_ITEM fn std::ptr::drop_in_place::<char> - shim(None) @@ unsizing-cgu.0[Internal]
    //~ MONO_ITEM fn <char as Trait>::foo
    let _char_unsized = char_sized as &Trait;

    // struct field
    let struct_sized = &Struct { _a: 1, _b: 2, _c: 3.0f64 };
    //~ MONO_ITEM fn std::ptr::drop_in_place::<f64> - shim(None) @@ unsizing-cgu.0[Internal]
    //~ MONO_ITEM fn <f64 as Trait>::foo
    let _struct_unsized = struct_sized as &Struct<Trait>;

    // custom coercion
    let wrapper_sized = Wrapper(&0u32);
    //~ MONO_ITEM fn std::ptr::drop_in_place::<u32> - shim(None) @@ unsizing-cgu.0[Internal]
    //~ MONO_ITEM fn <u32 as Trait>::foo
    let _wrapper_sized = wrapper_sized as Wrapper<Trait>;

    false.foo();

    0
}
