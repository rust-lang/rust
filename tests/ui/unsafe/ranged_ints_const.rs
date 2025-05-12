#![feature(rustc_attrs)]

#[rustc_layout_scalar_valid_range_start(1)]
#[repr(transparent)]
pub(crate) struct NonZero<T>(pub(crate) T);
fn main() {}

const fn foo() -> NonZero<u32> { NonZero(0) }
//~^ ERROR initializing type with `rustc_layout_scalar_valid_range` attr is unsafe

const fn bar() -> NonZero<u32> { unsafe { NonZero(0) } }
