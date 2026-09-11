//@ revisions: nogate gate
//@ [gate] check-fail
// FIXME(generic_const_parameter_types): this should pass
#![expect(incomplete_features)]
#![feature(adt_const_params, unsized_const_params, min_generic_const_args, generic_const_items)]
#![cfg_attr(gate, feature(generic_const_parameter_types))]

const FOO<T: core::marker::ConstParamTy_>: [T; 0] = core::direct_const_arg!(const { [] });
//~^ ERROR anonymous constants referencing generics are not yet supported

const BAR<const N: usize>: [(); N] = core::direct_const_arg!(const { [] });
//~^ ERROR anonymous constants referencing generics are not yet supported

const BAZ<'a>: [&'a (); 0] = core::direct_const_arg!(const { [] });
//~^ ERROR anonymous constants with lifetimes in their type are not yet supported

trait Tr {
    // FIXME(min_generic_const_args): These should error under [nogate]
    #[rustc_always_gca]
    const ASSOC<T: core::marker::ConstParamTy_>: [T; 0];

    #[rustc_always_gca]
    const ASSOC_CONST<const N: usize>: [(); N];

    #[rustc_always_gca]
    const ASSOC_LT<'a>: [&'a (); 0];
}

impl Tr for () {
    const ASSOC<T: core::marker::ConstParamTy_>: [T; 0] = core::direct_const_arg!(const { [] });
    //~^ ERROR anonymous constants referencing generics are not yet supported

    const ASSOC_CONST<const N: usize>: [(); N] = core::direct_const_arg!(const { [] });
    //~^ ERROR anonymous constants referencing generics are not yet supported

    const ASSOC_LT<'a>: [&'a (); 0] = core::direct_const_arg!(const { [] });
    //~^ ERROR anonymous constants with lifetimes in their type are not yet supported
}

fn main() {}
