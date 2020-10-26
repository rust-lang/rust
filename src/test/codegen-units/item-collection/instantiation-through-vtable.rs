// ignore-tidy-linelength
// compile-flags:-Zprint-mono-items=eager -Zinline-in-all-cgus -Zmir-opt-level=0

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

//~ MONO_ITEM fn start
#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    let s1 = Struct { _a: 0u32 };

    //~ MONO_ITEM fn std::intrinsics::drop_in_place::<Struct<u32>> - shim(None) @@ instantiation_through_vtable-cgu.0[Internal]
    //~ MONO_ITEM fn <Struct<u32> as Trait>::foo
    //~ MONO_ITEM fn <Struct<u32> as Trait>::bar
    let _ = &s1 as &Trait;

    let s1 = Struct { _a: 0u64 };
    //~ MONO_ITEM fn std::intrinsics::drop_in_place::<Struct<u64>> - shim(None) @@ instantiation_through_vtable-cgu.0[Internal]
    //~ MONO_ITEM fn <Struct<u64> as Trait>::foo
    //~ MONO_ITEM fn <Struct<u64> as Trait>::bar
    let _ = &s1 as &Trait;

    0
}
