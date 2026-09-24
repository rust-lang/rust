//@ check-pass

#![feature(min_generic_const_args)]
#![feature(adt_const_params, unsized_const_params)]

use std::gca;

#[derive(PartialEq, Eq, std::marker::ConstParamTy)]
enum Enum<T> {
    Unit,
    Tuple(),
    Store(T),
}

const _: Enum<()> = gca!(Enum::<()>::Unit);
const _: Enum<()> = gca!(Enum::<()>::Tuple());

fn main() {}
