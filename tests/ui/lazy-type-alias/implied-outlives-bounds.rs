// Check that we imply outlives-bounds on lazy type aliases.

//@ revisions: pos neg
//@[pos] check-pass

#![feature(lazy_type_alias)]
#![allow(incomplete_features)]

type TypeOutlives<'a, T> = &'a T;
type RegionOutlives<'a, 'b> = &'a &'b ();

// Ensure that we imply bounds from the explicit bounds of weak aliases.
struct Outer0<'a, T>(ExplicitTypeOutlives<'a, T>);
type ExplicitTypeOutlives<'a, T: 'a> = (&'a (), T);

// Ensure that we imply bounds from the implied bounds of weak aliases.
type Outer1<'b, U> = TypeOutlives<'b, U>;

#[cfg(neg)]
fn env0<'any>() {
    let _: TypeOutlives<'static, &'any ()>; //[neg]~ ERROR lifetime may not live long enough
}

#[cfg(neg)]
fn env1<'any>() {
    let _: RegionOutlives<'static, 'any>; //[neg]~ ERROR lifetime may not live long enough
}

#[cfg(neg)]
fn env2<'any>() {
    let _: Outer0<'static, &'any ()>; //[neg]~ ERROR lifetime may not live long enough
}

#[cfg(neg)]
fn env3<'any>() {
    let _: Outer1<'static, &'any ()>; //[neg]~ ERROR lifetime may not live long enough
}

fn main() {}
