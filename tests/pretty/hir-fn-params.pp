#[attr = MacroUse {arguments: UseAll}]
extern crate std;
#[prelude_import]
use ::std::prelude::rust_2015::*;
//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:hir-fn-params.pp

// This tests the pretty-printing of various kinds of function parameters.

//---------------------------------------------------------------------------
// Normal functions and methods.

fn normal_fn(_: u32, a: u32) { }

struct S;
impl S {
    fn method(_: u32, a: u32) { }
}

//---------------------------------------------------------------------------
// More exotic forms, which get a different pretty-printing path. In the past,
// anonymous params and `_` params printed incorrectly, e.g. `fn(u32, _: u32)`
// was printed as `fn(: u32, : u32)`.
//
// Ideally we would also test invalid patterns, e.g. `fn(1: u32, &a: u32)`,
// because they had similar problems. But the pretty-printing tests currently
// can't contain compile errors.

fn bare_fn(x: fn(u32, _: u32, a: u32)) { }

extern "C" {
    unsafe fn foreign_fn(_: u32, a: u32);
}

trait T {
    fn trait_fn(u32, _: u32, a: u32);
}
