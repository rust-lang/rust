// compile-flags:-Zprint-mono-items=eager -Zpolymorphize=on

#![feature(start)]

pub static FN : fn() = foo::<i32>;

pub fn foo<T>() { }

//~ MONO_ITEM fn foo::<T>
//~ MONO_ITEM static FN

//~ MONO_ITEM fn start
#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    0
}
