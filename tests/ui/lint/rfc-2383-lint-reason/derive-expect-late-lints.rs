// Make sure `#[expect]` on an item with a derive is fulfilled when the lint is only emitted on the
// item itself, and not also on the derived impl, which reuses the spans of the input.
//
// See <https://github.com/rust-lang/rust/issues/150553> for rationale.

//@ check-pass

#![feature(trivial_bounds)]
#![deny(drop_bounds, dyn_drop, trivial_bounds, non_upper_case_globals)]

#[derive(Clone)]
#[expect(drop_bounds)]
pub struct DropBound<T: Drop>(T);

#[derive(Clone, Copy)]
#[expect(dyn_drop)]
pub struct DynDrop(*const dyn Drop);

#[derive(Clone)]
#[expect(trivial_bounds)]
pub struct TrivialBound<T>(T)
where
    u8: Copy;

#[derive(Clone)]
#[expect(non_upper_case_globals)]
pub struct ConstParam<const n: usize>([u8; n]);

fn main() {}
