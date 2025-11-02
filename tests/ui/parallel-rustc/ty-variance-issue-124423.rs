// Test for #124423, which causes an ice bug: only `variances_of` returns `&[ty::Variance]`
//
//@ compile-flags: -Z threads=16
//@ compare-output-by-lines

use std::fmt::Debug;

fn elided(_: &impl Copy + 'a) -> _ { x }
//~^ ERROR ambiguous `+` in a type
//~| ERROR use of undeclared lifetime name `'a`
//~| ERROR the placeholder `_` is not allowed within types on item signatures for return types

fn explicit<'b>(_: &'a impl Copy + 'a) -> impl 'a { x }
//~^ ERROR ambiguous `+` in a type
//~| ERROR at least one trait must be specified
//~| ERROR use of undeclared lifetime name `'a`
//~| ERROR use of undeclared lifetime name `'a`
//~| ERROR use of undeclared lifetime name `'a`

fn elided2( impl 'b) -> impl 'a + 'a { x }
//~^ ERROR expected one of `:` or `|`, found `'b`
//~| ERROR expected identifier, found keyword `impl`
//~| ERROR at least one trait must be specified
//~| ERROR use of undeclared lifetime name `'a`
//~| ERROR use of undeclared lifetime name `'a`

fn explicit2<'a>(_: &'a impl Copy + 'a) -> impl Copy + 'a { x }
//~^ ERROR ambiguous `+` in a type

fn foo<'a>(_: &impl Copy + 'a) -> impl 'b + 'a { x }
//~^ ERROR ambiguous `+` in a type
//~| ERROR at least one trait must be specified
//~| ERROR use of undeclared lifetime name `'b`

fn elided3(_: &impl Copy + 'a) -> Box<dyn 'a> { Box::new(x) }
//~^ ERROR ambiguous `+` in a type
//~| ERROR use of undeclared lifetime name `'a`
//~| ERROR use of undeclared lifetime name `'a`
//~| ERROR at least one trait is required for an object type

fn x<'b>(_: &'a impl Copy + 'a) -> Box<dyn 'b> { Box::u32(x) }
//~^ ERROR ambiguous `+` in a type
//~| ERROR use of undeclared lifetime name `'a`
//~| ERROR use of undeclared lifetime name `'a`
//~| ERROR at least one trait is required for an object type
//~| ERROR no function or associated item named `u32` found for struct `Box<_, _>` in the current scope

fn elided4(_: &impl Copy + 'a) ->  new  { x(x) }
//~^ ERROR ambiguous `+` in a type
//~| ERROR use of undeclared lifetime name `'a`
//~| ERROR cannot find type `new` in this scope

trait LifetimeTrait<'a> {}

impl<'a> LifetimeTrait<'a> for &'a Box<dyn 'a> {}
//~^ ERROR at least one trait is required for an object type

fn main() {}
