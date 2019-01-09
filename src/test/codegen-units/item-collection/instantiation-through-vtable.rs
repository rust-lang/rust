// ignore-tidy-linelength
// compile-flags:-Zprint-mono-items=eager
// compile-flags:-Zinline-in-all-cgus

#![deny(dead_code)]
#![feature(start)]

trait Trait {
    fn foo(&self) -> u32;
    fn bar(&self);
}

struct Struct<T> {
    _a: T
}

impl<T> Trait for Struct<T> {
    fn foo(&self) -> u32 { 0 }
    fn bar(&self) {}
}

//~ MONO_ITEM fn instantiation_through_vtable::start[0]
#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    let s1 = Struct { _a: 0u32 };

    //~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<instantiation_through_vtable::Struct[0]<u32>> @@ instantiation_through_vtable-cgu.0[Internal]
    //~ MONO_ITEM fn instantiation_through_vtable::{{impl}}[0]::foo[0]<u32>
    //~ MONO_ITEM fn instantiation_through_vtable::{{impl}}[0]::bar[0]<u32>
    let _ = &s1 as &Trait;

    let s1 = Struct { _a: 0u64 };
    //~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<instantiation_through_vtable::Struct[0]<u64>> @@ instantiation_through_vtable-cgu.0[Internal]
    //~ MONO_ITEM fn instantiation_through_vtable::{{impl}}[0]::foo[0]<u64>
    //~ MONO_ITEM fn instantiation_through_vtable::{{impl}}[0]::bar[0]<u64>
    let _ = &s1 as &Trait;

    0
}
