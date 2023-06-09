//
// no-prefer-dynamic
// NOTE: We always compile this test with -Copt-level=0 because higher opt-levels
//       prevent drop-glue from participating in share-generics.
// incremental
// compile-flags:-Zprint-mono-items=eager -Zshare-generics=yes -Copt-level=0

#![crate_type="rlib"]

// aux-build:shared_generics_aux.rs
extern crate shared_generics_aux;

//~ MONO_ITEM fn foo
pub fn foo() {

    //~ MONO_ITEM fn shared_generics_aux::generic_fn::<u16> @@ shared_generics_aux-in-shared_generics.volatile[External]
    let _ = shared_generics_aux::generic_fn(0u16, 1u16);

    // This should not generate a monomorphization because it's already
    // available in `shared_generics_aux`.
    let _ = shared_generics_aux::generic_fn(0.0f32, 3.0f32);

    // The following line will drop an instance of `Foo`, generating a call to
    // Foo's drop-glue function. However, share-generics should take care of
    // reusing the drop-glue from the upstream crate, so we do not expect a
    // mono item for the drop-glue
    let _ = shared_generics_aux::Foo(1);
}
