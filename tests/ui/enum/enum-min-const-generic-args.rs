//@ run-pass

#![feature(min_generic_const_args)]
#![feature(adt_const_params, unsized_const_params)]

#[derive(PartialEq, Eq, std::marker::ConstParamTy)]
enum Enum<T> {
    Unit,
    Tuple(),
    Store(T),
}

// FIXME:  Ctor(Variant, Const)
type const _: Enum<()> = Enum::<()>::Unit;

// FIXME:  Ctor(Variant, Fn)
type const _: Enum<()> = Enum::<()>::Tuple();

// OK:  Variant
type const _: Enum<()> = Enum::<()>::Unit {};


type const _: Enum<()> = Enum::Unit::<()>; // OK
type const _: Enum<()> = Enum::Tuple::<()>(); // OK

type const _: Enum<()> = const { Enum::<()>::Unit }; // (OK)
type const _: Enum<()> = const { Enum::<()>::Tuple() }; // (OK)

fn main() {}
