//@ check-pass

#![feature(
    generic_const_items,
    min_generic_const_args,
    adt_const_params,
    generic_const_parameter_types,
    unsized_const_params,
)]
#![expect(incomplete_features)]

use std::marker::{PhantomData, ConstParamTy, ConstParamTy_};

#[derive(PartialEq, Eq, ConstParamTy)]
struct Foo<T> {
    field: T,
}

#[type_const]
const WRAP<T: ConstParamTy_, const N: T>: Foo<T> = { Foo::<T> {
    field: N,
} };

fn main() {
    // What we're trying to accomplish here is winding up with an equality relation
    // between two `ty::Const` that looks something like:
    //
    // ```
    // Foo<u8> { field: const { 1 + 2 } }
    // eq
    // Foo<u8> { field: ?x }
    // ```
    //
    // Note that the `field: _` here means a const argument `_` not a wildcard pattern.
    // This tests that we are able to infer `?x=3` even though the first `ty::Const`
    // may be a fully evaluated constant, and the latter is not fully evaluatable due
    // to inference variables.
    let _: PC<_, { WRAP::<u8, const { 1 + 1 }> }>
        =  PC::<_, { Foo::<u8> { field: _ }}>;
}

// "PhantomConst" helper equivalent to "PhantomData" used for testing equalities
// of arbitrarily typed const arguments.
struct PC<T: ConstParamTy_, const N: T> { _0: PhantomData<T> }
const PC<T: ConstParamTy_, const N: T>: PC<T, N> = PC { _0: PhantomData::<T> };
