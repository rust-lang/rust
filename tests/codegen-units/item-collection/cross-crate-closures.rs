// We need to disable MIR inlining in both this and its aux-build crate. The MIR inliner
// will just inline everything into our start function if we let it. As it should.
//@ compile-flags:-Clink-dead-code -Zinline-mir=no

#![deny(dead_code)]
#![no_main]

//@ aux-build:cgu_extern_closures.rs
extern crate cgu_extern_closures;

//~ MONO_ITEM fn main @@ cross_crate_closures-cgu.0[External]
#[no_mangle]
extern "C" fn main(_: core::ffi::c_int, _: *const *const u8) -> core::ffi::c_int {
    //~ MONO_ITEM fn cgu_extern_closures::inlined_fn @@ cross_crate_closures-cgu.0[External]
    //~ MONO_ITEM fn cgu_extern_closures::inlined_fn::{closure#0} @@ cross_crate_closures-cgu.0[External]
    let _ = cgu_extern_closures::inlined_fn(1, 2);

    //~ MONO_ITEM fn cgu_extern_closures::inlined_fn_generic::<i32> @@ cross_crate_closures-cgu.0[External]
    //~ MONO_ITEM fn cgu_extern_closures::inlined_fn_generic::<i32>::{closure#0} @@ cross_crate_closures-cgu.0[External]
    let _ = cgu_extern_closures::inlined_fn_generic(3, 4, 5i32);

    // Nothing should be generated for this call, we just link to the instance
    // in the extern crate.
    let _ = cgu_extern_closures::non_inlined_fn(6, 7);

    0
}
