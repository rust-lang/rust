// Check that we imply outlives-bounds on free const items.
//@ revisions: pos neg
//@[pos] check-pass
#![feature(generic_const_items)]
#![feature(freeze)] // only used in the test case `TYPE_OUTLIVES_1`
#![expect(incomplete_features)]

const REGION_OUTLIVES_0<'a, 'b>: Option<&'a &'b ()> = None; // we imply `'a: 'b`
const REGION_OUTLIVES_1<'a, 'b>: &'a &'b () = &&(); // we imply `'a: 'b`

const TYPE_OUTLIVES_0<'a, T>: Option<&'a T> = None; // we imply `T: 'a`

const TYPE_OUTLIVES_1<'a, T: Def>: &'a T = &T::DEF; // we imply `T: 'a`
trait Def: std::marker::Freeze { const DEF: Self; }

// Ensure that we actually enforce these implied bounds at usage sites:

#[cfg(neg)]
fn env0<'any>() {
    _ = TYPE_OUTLIVES_0::<'static, &'any ()>; //[neg]~ ERROR lifetime may not live long enough
}

#[cfg(neg)]
fn env1<'any>() {
    _ = REGION_OUTLIVES_0::<'static, 'any>; //[neg]~ ERROR lifetime may not live long enough
}

fn main() {}
