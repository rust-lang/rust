#![feature(adt_const_params)]

#[derive(Eq, PartialEq, std::marker::ConstParamTy)]
struct Inner<const N: usize>;

struct Foo<
    const PARAM_TY: Inner<const { 1 }>,
    //~^ ERROR: unbraced const blocks as const args are experimental
    const DEFAULT: usize = const { 1 },
    //~^ ERROR: unbraced const blocks as const args are experimental
>;

type Array = [(); const { 1 }];
type NormalTy = Inner<const { 1 }>;
//~^ ERROR: unbraced const blocks as const args are experimental

fn repeat() {
    [1_u8; const { 1 }];
}

fn body_ty() {
    let _: Inner<const { 1 }>;
    //~^ ERROR: unbraced const blocks as const args are experimental
}

fn generic<const N: usize>() {
    if false {
        generic::<const { 1 }>();
        //~^ ERROR: unbraced const blocks as const args are experimental
    }
}

const NON_TYPE_CONST: usize = const { 1 };

const TYPE_CONST: usize = core::direct_const_arg!(const { 1 });
//~^ ERROR: use of unstable library feature `min_generic_const_args` [E0658]
//~| ERROR: expected expression, found `direct_const_arg!()` constant

static STATIC: usize = const { 1 };

fn main() {}
