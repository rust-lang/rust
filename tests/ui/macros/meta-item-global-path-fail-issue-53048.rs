// Regression test for https://github.com/rust-lang/rust/issues/53048/
// This test ensures that invalid meta item or global path
// is rejected by the compiler.

extern crate core;

macro_rules! m { ($m:meta) => { #[derive($m)] pub struct S; }; }

m!(a(::b::c));
//~^ ERROR traits in `#[derive(...)]` don't accept arguments

m!(::);
//~^ ERROR expected identifier, found `<eof>`

// a meta item must begin with a path
m!(1 + 1);
//~^ ERROR no rules expected `1`

// must be a simple path, no generic arguments allowed
m!(a::<b>::c);
//~^ ERROR expected identifier, found `<`

// must be a simple path, no qualified self types allowed
m!(<T>::foo);
//~^ ERROR no rules expected `<`

m!(<T as a>::foo);
//~^ ERROR no rules expected `<`

// no segment after `::` to be a valid path
#[derive(::)] pub struct D;
//~^ ERROR expected unsuffixed literal, found `<eof>`

fn main() {}
