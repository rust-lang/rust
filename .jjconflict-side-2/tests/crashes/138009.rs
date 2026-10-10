//@ known-bug: #138009
#![feature(gca_min_const_items)]

use std::gca;

#[repr(simd)]
struct T([isize; gca!(N)]);

static X: T = T();
