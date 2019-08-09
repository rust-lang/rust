// In the current version of the collector that still has to support
// legacy-codegen, closures do not generate their own MonoItems, so we are
// ignoring this test until MIR codegen has taken over completely
// ignore-test

// compile-flags:-Zprint-mono-items=eager

#![deny(dead_code)]
#![feature(start)]

// aux-build:cgu_extern_closures.rs
extern crate cgu_extern_closures;

//~ MONO_ITEM fn cross_crate_closures::start[0]
#[start]
fn start(_: isize, _: *const *const u8) -> isize {

    //~ MONO_ITEM fn cgu_extern_closures::inlined_fn[0]
    //~ MONO_ITEM fn cgu_extern_closures::inlined_fn[0]::{{closure}}[0]
    let _ = cgu_extern_closures::inlined_fn(1, 2);

    //~ MONO_ITEM fn cgu_extern_closures::inlined_fn_generic[0]<i32>
    //~ MONO_ITEM fn cgu_extern_closures::inlined_fn_generic[0]::{{closure}}[0]<i32>
    let _ = cgu_extern_closures::inlined_fn_generic(3, 4, 5i32);

    // Nothing should be generated for this call, we just link to the instance
    // in the extern crate.
    let _ = cgu_extern_closures::non_inlined_fn(6, 7);

    0
}

//~ MONO_ITEM drop-glue i8
