// Test for #124423, which causes an ice bug
//
//@ parallel-front-end-robustness
//@ compile-flags: -Z threads=16

use std::fmt::Debug;

fn elided(_: &impl Copy + 'a) -> _ { x }
fn explicit<'b>(_: &'a impl Copy + 'a) -> impl 'a { x }
fn elided2( impl 'b) -> impl 'a + 'a { x }
fn explicit2<'a>(_: &'a impl Copy + 'a) -> impl Copy + 'a { x }
fn foo<'a>(_: &impl Copy + 'a) -> impl 'b + 'a { x }
fn elided3(_: &impl Copy + 'a) -> Box<dyn 'a> { Box::new(x) }
fn x<'b>(_: &'a impl Copy + 'a) -> Box<dyn 'b> { Box::u32(x) }
fn elided4(_: &impl Copy + 'a) ->  new  { x(x) }
trait LifetimeTrait<'a> {}
impl<'a> LifetimeTrait<'a> for &'a Box<dyn 'a> {}

fn main() {}
