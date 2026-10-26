//@ check-pass

#![feature(adt_const_params, gca_adts, gca_min_const_items, unsized_const_params)]

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
