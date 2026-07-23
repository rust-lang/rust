// Regression test for #132986.
// FIXME(min_generic_const_args): using statics as direct const arguments should error instead of
// ICEing until const eval can evaluate statics to valtrees for const generics.

#![feature(min_generic_const_args, macroless_generic_const_args)]
#![allow(incomplete_features)]

static A: u32 = 0;

struct Foo<const N: u32>;

const _: Foo<{ A }> = Foo;
//~^ ERROR static items cannot be used as const arguments

const _: Foo<A> = Foo;
//~^ ERROR static items cannot be used as const arguments

fn main() {}
