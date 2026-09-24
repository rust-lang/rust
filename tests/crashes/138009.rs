//@ known-bug: #138009
#![feature(min_generic_const_args)]

use std::gca;

#[repr(simd)]
struct T([isize; gca!(N)]);

static X: T = T();
