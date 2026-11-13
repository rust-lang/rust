// Check that we imply outlives-bounds on checked type aliases.

//@ revisions: pass fail dump
//@[pass] check-pass

#![feature(checked_type_aliases)]
#![cfg_attr(dump, feature(rustc_attrs))]

#[cfg_attr(dump, rustc_dump_inferred_outlives)]
type TypeOutlives<'a, T> = &'a T;
//[dump]~^ ERROR rustc_dump_inferred_outlives
//[dump]~| NOTE T: 'a

#[cfg_attr(dump, rustc_dump_inferred_outlives)]
type RegionOutlives<'a, 'b> = &'a &'b ();
//[dump]~^ ERROR rustc_dump_inferred_outlives
//[dump]~| NOTE 'b: 'a

// Ensure that we imply bounds from the explicit bounds of free alias types.
#[cfg_attr(dump, rustc_dump_inferred_outlives)]
struct Outer0<'a, T>(ExplicitTypeOutlives<'a, T>);
//[dump]~^ ERROR rustc_dump_inferred_outlives
//[dump]~| NOTE T: 'a

type ExplicitTypeOutlives<'a, T: 'a> = (&'a (), T);

// Ensure that we imply bounds from the implied bounds of free alias types.
#[cfg_attr(dump, rustc_dump_inferred_outlives)]
type Outer1<'b, U> = TypeOutlives<'b, U>;
//[dump]~^ ERROR rustc_dump_inferred_outlives
//[dump]~| NOTE U: 'b

#[cfg(fail)] fn env0<'any>() { let _: TypeOutlives<'static, &'any ()>; }
//[fail]~^ ERROR lifetime may not live long enough

#[cfg(fail)] fn env1<'any>() { let _: RegionOutlives<'static, 'any>; }
//[fail]~^ ERROR lifetime may not live long enough

#[cfg(fail)] fn env2<'any>() { let _: Outer0<'static, &'any ()>; }
//[fail]~^ ERROR lifetime may not live long enough

#[cfg(fail)] fn env3<'any>() { let _: Outer1<'static, &'any ()>; }
//[fail]~^ ERROR lifetime may not live long enough

fn main() {}
