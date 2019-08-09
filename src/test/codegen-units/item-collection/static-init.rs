// compile-flags:-Zprint-mono-items=eager

#![feature(start)]

pub static FN : fn() = foo::<i32>;

pub fn foo<T>() { }

//~ MONO_ITEM fn static_init::foo[0]<i32>
//~ MONO_ITEM static static_init::FN[0]

//~ MONO_ITEM fn static_init::start[0]
#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    0
}
