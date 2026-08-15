//@ compile-flags:-Clink-dead-code

#![deny(dead_code)]
#![crate_type = "lib"]

//@ aux-build:cgu_generic_function.rs
extern crate cgu_generic_function;

//~ MONO_ITEM fn start
#[no_mangle]
pub fn start(_: isize, _: *const *const u8) -> isize {
    //~ MONO_ITEM fn cgu_generic_function::bar::<u32>
    //~ MONO_ITEM fn cgu_generic_function::foo::<u32>
    let _ = cgu_generic_function::foo(1u32);

    //~ MONO_ITEM fn cgu_generic_function::bar::<u64>
    //~ MONO_ITEM fn cgu_generic_function::foo::<u64>
    let _ = cgu_generic_function::foo(2u64);

    // This should not introduce a codegen item
    let _ = cgu_generic_function::exported_but_not_generic(3);

    0
}
