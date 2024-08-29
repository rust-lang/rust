// We need to disable MIR inlining in both this and its aux-build crate. The MIR inliner
// will just inline everything into our start function if we let it. As it should.
//@ compile-flags:-Zprint-mono-items=eager -Zinline-mir=no

#![deny(dead_code)]
#![feature(start)]

//@ aux-build:cgu_extern_closures.rs
extern crate cgu_extern_closures;

//~ MONO_ITEM fn start @@ cross_crate_closures-cgu.0[Internal]
#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    //~ MONO_ITEM fn cgu_extern_closures::inlined_fn @@ cross_crate_closures-cgu.0[Internal]
    //~ MONO_ITEM fn cgu_extern_closures::inlined_fn::{closure#0} @@ cross_crate_closures-cgu.0[Internal]
    let _ = cgu_extern_closures::inlined_fn(1, 2);

    //~ MONO_ITEM fn cgu_extern_closures::inlined_fn_generic::<i32> @@ cross_crate_closures-cgu.0[Internal]
    //~ MONO_ITEM fn cgu_extern_closures::inlined_fn_generic::<i32>::{closure#0} @@ cross_crate_closures-cgu.0[Internal]
    let _ = cgu_extern_closures::inlined_fn_generic(3, 4, 5i32);

    // Nothing should be generated for this call, we just link to the instance
    // in the extern crate.
    let _ = cgu_extern_closures::non_inlined_fn(6, 7);

    0
}
