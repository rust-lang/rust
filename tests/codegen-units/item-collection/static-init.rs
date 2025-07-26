//@ compile-flags:-Clink-dead-code

#![crate_type = "lib"]

static FN: fn() = foo::<i32>;

fn foo<T>() {}

//~ MONO_ITEM fn foo::<i32>
//~ MONO_ITEM static FN

//~ MONO_ITEM fn start
#[no_mangle]
pub fn start(_: isize, _: *const *const u8) -> isize {
    0
}
