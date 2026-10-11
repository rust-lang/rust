// Regression test for #132986.
// FIXME(gca_min_const_items): using statics as direct const arguments should error instead of
// ICEing until const eval can evaluate statics to valtrees for const generics.

#![feature(gca_min_const_items)]
#![allow(incomplete_features)]

use std::gca;

static A: u32 = 0;

struct Foo<const N: u32>;

const _: Foo<{ gca!(A) }> = Foo;
//~^ ERROR static items cannot be used as const arguments

const _: Foo<gca!(A)> = Foo;
//~^ ERROR static items cannot be used as const arguments

fn main() {}
