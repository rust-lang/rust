#![deny(single_use_lifetimes)]
#![allow(dead_code)]
#![allow(unused_variables)]

// Test that we DO warn when lifetime name is used only
// once in a fn argument.

fn a<'a>(x: &'a u32) { //~ ERROR `'a` only used once
    //~^ HELP elide the single-use lifetime
}

struct Single<'a> { x: &'a u32 }
struct Double<'a, 'b> { f: &'a &'b u32 }

fn center<'m>(_: Single<'m>) {} //~ ERROR `'m` only used once
//~^ HELP elide the single-use lifetime
fn left<'x, 'y>(foo: Double<'x, 'y>) -> &'x u32 { foo.f } //~ ERROR `'y` only used once
//~^ HELP elide the single-use lifetime
fn right<'x, 'y>(foo: Double<'x, 'y>) -> &'y u32 { foo.f } //~ ERROR `'x` only used once
//~^ HELP elide the single-use lifetime

pub trait Tfv<'a> {}

// Do NOT lint in an HRTB.
pub fn g<T: for<'a> Tfv<'a>>() {}

// Do NOT lint for trait bounds.
pub fn h<'a, S>(_: S)
where
    S: Tfv<'a>,
{}

fn main() {}
