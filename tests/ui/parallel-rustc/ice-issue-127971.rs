// Test for #127971, which causes an ice bug
//
//@ parallel-front-end-robustness
//@ compile-flags: -Z threads=16

use std::fmt::Debug;

fn elided(_: &impl Copy + 'a) -> _ { x }

fn foo<'a>(_: &impl Copy + 'a) -> impl 'b + 'a { x }

fn x<'b>(_: &'a impl Copy + 'a) -> Box<dyn 'b> { Box::u32(x) }

fn main() {}
