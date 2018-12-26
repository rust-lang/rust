// ignore-tidy-linelength
// compile-flags:-Zprint-mono-items=eager -Zshare-generics=yes -Zincremental=tmp/partitioning-tests/shared-generics-exe

#![crate_type="rlib"]

// aux-build:shared_generics_aux.rs
extern crate shared_generics_aux;

//~ MONO_ITEM fn shared_generics::foo[0]
pub fn foo() {

    //~ MONO_ITEM fn shared_generics_aux::generic_fn[0]<u16> @@ shared_generics_aux-in-shared_generics.volatile[External]
    let _ = shared_generics_aux::generic_fn(0u16, 1u16);

    // This should not generate a monomorphization because it's already
    // available in `shared_generics_aux`.
    let _ = shared_generics_aux::generic_fn(0.0f32, 3.0f32);
}

// MONO_ITEM drop-glue i8
