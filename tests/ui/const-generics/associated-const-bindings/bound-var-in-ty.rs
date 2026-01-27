// Check that we don't reject non-escaping late-bound vars in the type of assoc const bindings.
// There's no reason why we should disallow them.
//
//@ check-pass

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
        <for<'a> fn(&'a str) -> &'a str as Discard>::Out,
        K = const { }
    >,
) {}

trait Discard { type Out; }
impl<T: ?Sized> Discard for T { type Out = (); }

fn main() {}
