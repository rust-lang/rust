//@ parallel-front-end
//@ compile-flags: -Z threads=16

use std::fmt::Debug;

fn elided(_: &impl Copy + 'a) -> _ { x }
//~^ ERROR
//~| ERROR
//~| ERROR

fn foo<'a>(_: &impl Copy + 'a) -> impl 'b + 'a { x }
//~^ ERROR
//~| ERROR
//~| ERROR

fn x<'b>(_: &'a impl Copy + 'a) -> Box<dyn 'b> { Box::u32(x) }
//~^ ERROR
//~| ERROR
//~| ERROR
//~| ERROR
//~| ERROR

fn main() {}
