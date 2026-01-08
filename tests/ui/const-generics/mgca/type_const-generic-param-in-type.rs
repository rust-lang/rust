//@ revisions: nogate gate
//@ [gate] check-fail
// FIXME(generic_const_parameter_types): this should pass
#![expect(incomplete_features)]
#![feature(adt_const_params, unsized_const_params, min_generic_const_args, generic_const_items)]
#![cfg_attr(gate, feature(generic_const_parameter_types))]

#[type_const]
const FOO<T: core::marker::ConstParamTy_>: [T; 0] = const { [] };
//[nogate]~^ ERROR the type of const parameters must not depend on other generic parameters
//[gate]~^^ ERROR anonymous constants referencing generics are not yet supported

#[type_const]
const BAR<const N: usize>: [(); N] = const { [] };
//[nogate]~^ ERROR the type of const parameters must not depend on other generic parameters
//[gate]~^^ ERROR anonymous constants referencing generics are not yet supported

#[type_const]
const BAZ<'a>: [&'a (); 0] = const { [] };
//[nogate]~^ ERROR the type of const parameters must not depend on other generic parameters
//[gate]~^^ ERROR anonymous constants with lifetimes in their type are not yet supported

trait Tr {
    #[type_const]
    const ASSOC<T: core::marker::ConstParamTy_>: [T; 0];
    //[nogate]~^ ERROR the type of const parameters must not depend on other generic parameters

    #[type_const]
    const ASSOC_CONST<const N: usize>: [(); N];
    //[nogate]~^ ERROR the type of const parameters must not depend on other generic parameters

    #[type_const]
    const ASSOC_LT<'a>: [&'a (); 0];
    //[nogate]~^ ERROR the type of const parameters must not depend on other generic parameters
}

impl Tr for () {
    #[type_const]
    const ASSOC<T: core::marker::ConstParamTy_>: [T; 0] = const { [] };
    //[nogate]~^ ERROR the type of const parameters must not depend on other generic parameters
    //[gate]~^^ ERROR anonymous constants referencing generics are not yet supported

    #[type_const]
    const ASSOC_CONST<const N: usize>: [(); N] = const { [] };
    //[nogate]~^ ERROR the type of const parameters must not depend on other generic parameters
    //[gate]~^^ ERROR anonymous constants referencing generics are not yet supported

    #[type_const]
    const ASSOC_LT<'a>: [&'a (); 0] = const { [] };
    //[nogate]~^ ERROR the type of const parameters must not depend on other generic parameters
    //[gate]~^^ ERROR anonymous constants with lifetimes in their type are not yet supported
}

fn main() {}
