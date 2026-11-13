// Check that we imply outlives-bounds on (generic) free const items.
//@ revisions: pass fail dump
//@[pass] check-pass
#![feature(generic_const_items)]
#![feature(freeze)] // only used in the test case `TYPE_OUTLIVES_1`
#![cfg_attr(dump, feature(rustc_attrs))]

// We imply `'b: 'a` from type `&'a &'b ()` in the constant's type.
#[cfg_attr(dump, rustc_dump_inferred_outlives)]
const REGION_OUTLIVES_0<'a, 'b>: Option<&'a &'b ()> = None;
//[dump]~^ ERROR rustc_dump_inferred_outlives
//[dump]~| NOTE: 'b: 'a

// We imply `'b: 'a` from type `&'a &'b ()` in the constant's type.
#[cfg_attr(dump, rustc_dump_inferred_outlives)]
const REGION_OUTLIVES_1<'a, 'b>: &'a &'b () = &&();
//[dump]~^ ERROR rustc_dump_inferred_outlives
//[dump]~| NOTE: 'b: 'a

// We imply `T: 'a` from type `&'a T` in the constant's type.
#[cfg_attr(dump, rustc_dump_inferred_outlives)]
const TYPE_OUTLIVES_0<'a, T>: Option<&'a T> = None;
//[dump]~^ ERROR rustc_dump_inferred_outlives
//[dump]~| NOTE: T: 'a

// We imply `T: 'a` from type `&'a T` in the constant's type.
#[cfg_attr(dump, rustc_dump_inferred_outlives)]
const TYPE_OUTLIVES_1<'a, T: Def>: &'a T = &T::DEF;
//[dump]~^ ERROR rustc_dump_inferred_outlives
//[dump]~| NOTE: T: 'a

trait Def: std::marker::Freeze { const DEF: Self; }

// Ensure that we actually enforce these implied bounds at usage sites:

#[cfg(fail)] fn env0<'any>() { let _ = REGION_OUTLIVES_0::<'static, 'any>; }
//[fail]~^ ERROR lifetime may not live long enough

#[cfg(fail)] fn env1<'any>() { let _ = TYPE_OUTLIVES_0::<'static, &'any ()>; }
//[fail]~^ ERROR lifetime may not live long enough

fn main() {}
