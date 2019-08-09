// compile-flags:-Zprint-mono-items=eager
// compile-flags:-Zinline-in-all-cgus

#![deny(dead_code)]
#![feature(coerce_unsized)]
#![feature(unsize)]
#![feature(start)]

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
    _c: T
}

impl Trait for f64 {
    fn foo(&self) {}
}

// Custom Coercion Case
impl Trait for u32 {
    fn foo(&self) {}
}

#[derive(Clone, Copy)]
struct Wrapper<T: ?Sized>(*const T);

impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<Wrapper<U>> for Wrapper<T> {}

//~ MONO_ITEM fn unsizing::start[0]
#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    // simple case
    let bool_sized = &true;
    //~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<bool> @@ unsizing-cgu.0[Internal]
    //~ MONO_ITEM fn unsizing::{{impl}}[0]::foo[0]
    let _bool_unsized = bool_sized as &Trait;

    let char_sized = &'a';

    //~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<char> @@ unsizing-cgu.0[Internal]
    //~ MONO_ITEM fn unsizing::{{impl}}[1]::foo[0]
    let _char_unsized = char_sized as &Trait;

    // struct field
    let struct_sized = &Struct {
        _a: 1,
        _b: 2,
        _c: 3.0f64
    };
    //~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<f64> @@ unsizing-cgu.0[Internal]
    //~ MONO_ITEM fn unsizing::{{impl}}[2]::foo[0]
    let _struct_unsized = struct_sized as &Struct<Trait>;

    // custom coercion
    let wrapper_sized = Wrapper(&0u32);
    //~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<u32> @@ unsizing-cgu.0[Internal]
    //~ MONO_ITEM fn unsizing::{{impl}}[3]::foo[0]
    let _wrapper_sized = wrapper_sized as Wrapper<Trait>;

    0
}
