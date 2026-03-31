//! Check associated const binding with escaping bound vars doesn't cause ICE
//! (#151642)
//@ check-pass

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

trait Trait2<'a> { type const ASSOC: i32; }
fn g(_: for<'a> fn(Box<dyn Trait2<'a, ASSOC = 10>>)) {}

fn main() {}
