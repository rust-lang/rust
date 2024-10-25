//@ parallel-front-end
//@ compile-flags: -Z threads=16

use std::fmt::Debug;

fn elided(_: &impl Copy + 'a) -> _ { x }
//~^ ERROR
//~| ERROR
//~| ERROR
fn explicit<'b>(_: &'a impl Copy + 'a) -> impl 'a { x }
//~^ ERROR
//~| ERROR
//~| ERROR
//~| ERROR
//~| ERROR
fn elided2( impl 'b) -> impl 'a + 'a { x }
//~^ ERROR
//~| ERROR
//~| ERROR
//~| ERROR
//~| ERROR
fn explicit2<'a>(_: &'a impl Copy + 'a) -> impl Copy + 'a { x }
//~^ ERROR
fn foo<'a>(_: &impl Copy + 'a) -> impl 'b + 'a { x }
//~^ ERROR
//~| ERROR
//~| ERROR
fn elided3(_: &impl Copy + 'a) -> Box<dyn 'a> { Box::new(x) }
//~^ ERROR
//~| ERROR
//~| ERROR
//~| ERROR
fn x<'b>(_: &'a impl Copy + 'a) -> Box<dyn 'b> { Box::u32(x) }
//~^ ERROR
//~| ERROR
//~| ERROR
//~| ERROR
//~| ERROR
fn elided4(_: &impl Copy + 'a) ->  new  { x(x) }
//~^ ERROR
//~| ERROR
//~| ERROR
trait LifetimeTrait<'a> {}
impl<'a> LifetimeTrait<'a> for &'a Box<dyn 'a> {}
//~^ ERROR

fn main() {}
