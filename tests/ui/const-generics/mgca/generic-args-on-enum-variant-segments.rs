//@ check-pass

#![feature(min_generic_const_args)]
#![feature(adt_const_params, unsized_const_params)]

#[derive(PartialEq, Eq, std::marker::ConstParamTy)]
enum Enum<T> {
    Unit,
    Tuple(),
    Store(T),
}

const _: Enum<()> = core::direct_const_arg!(Enum::<()>::Unit);
const _: Enum<()> = core::direct_const_arg!(Enum::<()>::Tuple());

fn main() {}
