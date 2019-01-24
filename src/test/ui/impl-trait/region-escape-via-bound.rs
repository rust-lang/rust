// run-pass
//
// Test that we allow the region `'x` to escape in the impl
// because 'y` escapes, which outlives `'x`.
//
// See https://github.com/rust-lang/rust/issues/46541 for more details.

#![allow(dead_code)]
#![feature(in_band_lifetimes)]
#![feature(nll)]

use std::cell::Cell;

trait Trait<'a> { }

impl Trait<'b> for Cell<&'a u32> { }

fn foo(x: Cell<&'x u32>) -> impl Trait<'y>
    // ^ hidden type for `impl Trait` captures lifetime that does not appear in bounds
    // because it outlives the lifetime that *does* appear in the bounds, `'y`
where 'x: 'y
{
    x
}

fn main() { }
