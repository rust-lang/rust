#![feature(min_generic_const_args)]
#![allow(incomplete_features)]

pub const NON_LOCAL_CONST: char = core::direct_const_arg!('a');
