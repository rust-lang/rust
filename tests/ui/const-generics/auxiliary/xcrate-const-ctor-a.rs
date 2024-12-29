// NOTE: This aux file inherits revisions from its parent tests.

#![feature(adt_const_params)]
#![cfg_attr(mgca, feature(min_generic_const_args), allow(incomplete_features))]

use std::marker::ConstParamTy;

#[derive(Eq, PartialEq, ConstParamTy)]
pub struct Foo;
