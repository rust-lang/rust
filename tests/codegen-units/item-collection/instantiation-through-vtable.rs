//@ compile-flags:-Clink-dead-code -Zmir-opt-level=0

#![deny(dead_code)]
#![crate_type = "lib"]

trait Trait {
    fn foo(&self) -> u32;
    fn bar(&self);
}

struct Struct<T> {
    _a: T,
}

impl<T> Trait for Struct<T> {
    fn foo(&self) -> u32 {
        0
    }
    fn bar(&self) {}
}

//~ MONO_ITEM fn start
#[no_mangle]
pub fn start(_: isize, _: *const *const u8) -> isize {
    let s1 = Struct { _a: 0u32 };

    //~ MONO_ITEM fn <Struct<u32> as Trait>::foo
    //~ MONO_ITEM fn <Struct<u32> as Trait>::bar
    let r1 = &s1 as &Trait;
    r1.foo();
    r1.bar();

    let s1 = Struct { _a: 0u64 };
    //~ MONO_ITEM fn <Struct<u64> as Trait>::foo
    //~ MONO_ITEM fn <Struct<u64> as Trait>::bar
    let _ = &s1 as &Trait;

    0
}
