#![feature(rustc_attrs)]
#![crate_type = "lib"]

// Check that various `#[rustc_nonnull_optimization_guaranteed]` types
// get their expected layouts inside `Option`s.

use std::num::NonZero;

#[rustc_dump_layout(backend_repr)]
type OptNonZeroU8 = Option<NonZero<u8>>;
//~^ ERROR: Scalar(u8 is ..)

#[rustc_dump_layout(backend_repr)]
type OptNonZeroU16 = Option<NonZero<u16>>;
//~^ ERROR: Scalar(u16 is ..)

#[rustc_dump_layout(backend_repr)]
type OptNonZeroU32 = Option<NonZero<u32>>;
//~^ ERROR: Scalar(u32 is ..)

#[rustc_dump_layout(backend_repr)]
type OptNonZeroU64 = Option<NonZero<u64>>;
//~^ ERROR: Scalar(u64 is ..)

#[rustc_dump_layout(backend_repr)]
type OptNonZeroU128 = Option<NonZero<u128>>;
//~^ ERROR: Scalar(u128 is ..)

#[rustc_dump_layout(backend_repr)]
type OptNonZeroI8 = Option<NonZero<i8>>;
//~^ ERROR: Scalar(i8 is ..)

#[rustc_dump_layout(backend_repr)]
type OptNonZeroI16 = Option<NonZero<i16>>;
//~^ ERROR: Scalar(i16 is ..)

#[rustc_dump_layout(backend_repr)]
type OptNonZeroI32 = Option<NonZero<i32>>;
//~^ ERROR: Scalar(i32 is ..)

#[rustc_dump_layout(backend_repr)]
type OptNonZeroI64 = Option<NonZero<i64>>;
//~^ ERROR: Scalar(i64 is ..)

#[rustc_dump_layout(backend_repr)]
type OptNonZeroI128 = Option<NonZero<i128>>;
//~^ ERROR: Scalar(i128 is ..)
