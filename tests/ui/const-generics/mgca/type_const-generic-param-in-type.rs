//@ revisions: nogate gate
//@ [gate] check-pass
#![expect(incomplete_features)]
#![feature(adt_const_params, unsized_const_params, min_generic_const_args, generic_const_items)]
#![cfg_attr(gate, feature(generic_const_parameter_types))]

use std::gca;
use std::marker::ConstParamTy;

#[derive(ConstParamTy, PartialEq, Eq, Debug)]
struct StructWithConstParam<const N: usize>;

const FOO<T: core::marker::ConstParamTy_>: [T; 0] = gca!([]);
//[nogate]~^ ERROR the type of const parameters must not depend on other generic parameters

const BAR<const N: usize>: StructWithConstParam<N> = gca!(StructWithConstParam::<N>);
//[nogate]~^ ERROR the type of const parameters must not depend on other generic parameters

const BAZ<'a>: [&'a (); 0] = gca!([]);
//[nogate]~^ ERROR the type of const parameters must not depend on other generic parameters

trait Tr {
    #[rustc_always_gca]
    const ASSOC<T: core::marker::ConstParamTy_>: [T; 0];
    //[nogate]~^ ERROR the type of const parameters must not depend on other generic parameters

    #[rustc_always_gca]
    const ASSOC_CONST<const N: usize>: StructWithConstParam<N>;
    //[nogate]~^ ERROR the type of const parameters must not depend on other generic parameters

    #[rustc_always_gca]
    const ASSOC_LT<'a>: [&'a (); 0];
    //[nogate]~^ ERROR the type of const parameters must not depend on other generic parameters
}

impl Tr for () {
    const ASSOC<T: core::marker::ConstParamTy_>: [T; 0] = gca!([]);
    //[nogate]~^ ERROR the type of const parameters must not depend on other generic parameters

    const ASSOC_CONST<const N: usize>: StructWithConstParam<N> = gca!(StructWithConstParam::<N>);
    //[nogate]~^ ERROR the type of const parameters must not depend on other generic parameters

    const ASSOC_LT<'a>: [&'a (); 0] = gca!([]);
    //[nogate]~^ ERROR the type of const parameters must not depend on other generic parameters
}

fn main() {}
