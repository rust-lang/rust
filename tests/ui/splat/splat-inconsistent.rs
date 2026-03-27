//! Test using `#[splat]` incorrectly, in ways not covered by other tests.

#![allow(incomplete_features)]
#![feature(splat)]

// FIXME(splat): multiple splats in a fn should error.
// For now, the attribute is incorrectly assumed to be on the last argument.
fn multisplat_bad_2(#[splat] (_a, _b): (u32, i8), #[splat] (_c, _d): (u32, i8)) {}

// FIXME(splat): non-terminal splat attributes should error, until we have a specific use case for
// them.
// For now, the attribute is incorrectly assumed to be on the last argument.
fn splat_non_terminal_bad(#[splat] (_a, _b): (u32, i8), (_c, _d): (u32, i8)) {}

extern "C" {
    // FIXME(splat): tuple layouts are unspecified. Should this error in addition to
    // the existing `improper_ctypes` lint?
    #[expect(improper_ctypes)]
    fn bar_2(#[splat] _: (u32, i8));
}

trait FooTrait {
    fn has_splat(#[splat] _: ());

    fn no_splat(_: (u32, f64));
}

struct Foo;

impl FooTrait for Foo {
    fn has_splat(_: ()) {} //~ ERROR method `has_splat` has an incompatible type for trait

    fn no_splat(#[splat] _: (u32, f64)) {} //~ ERROR method `no_splat` has an incompatible type for trait
}

fn main() {}
