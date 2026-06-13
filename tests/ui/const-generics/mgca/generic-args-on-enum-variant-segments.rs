//@ check-pass

#![feature(min_generic_const_args)]
#![feature(adt_const_params, unsized_const_params)]

#[derive(PartialEq, Eq, std::marker::ConstParamTy)]
enum Enum<T> {
    Unit,
    Tuple(),
    Store(T),
}

type const _: Enum<()> = Enum::<()>::Unit;
type const _: Enum<()> = Enum::<()>::Tuple();

fn main() {}
