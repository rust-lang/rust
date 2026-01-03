// Check that we eventually catch types of assoc const bounds
// (containing late-bound vars) that are ill-formed.
#![feature(
    min_generic_const_args,
    adt_const_params,
    unsized_const_params,
    generic_const_parameter_types,
)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy_;

trait Trait<T: ConstParamTy_> {
    #[type_const]
    const K: T;
}

fn take(
    _: impl Trait<
        <<for<'a> fn(&'a str) -> &'a str as Project>::Out as Discard>::Out,
        K = const { () }
    >,
) {}
//~^^^ ERROR higher-ranked subtype error
//~| ERROR higher-ranked subtype error

trait Project { type Out; }
impl<T> Project for fn(T) -> T { type Out = T; }

trait Discard { type Out; }
impl<T: ?Sized> Discard for T { type Out = (); }

fn main() {}
