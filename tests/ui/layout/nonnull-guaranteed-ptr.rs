//@ only-64bit
#![feature(rustc_attrs)]
#![crate_type = "lib"]

// Check that various `#[rustc_nonnull_optimization_guaranteed]` types
// get their expected layouts inside `Option`s.

use std::ptr::NonNull;

#[rustc_dump_layout(backend_repr)]
type OptNonNull = Option<NonNull<String>>;
//~^ ERROR: Scalar(pointer is 0..=18446744073709551615)

#[rustc_dump_layout(backend_repr)]
type OptRef<'a> = Option<&'a String>;
//~^ ERROR: Scalar(pointer is 0..=18446744073709551615)

#[rustc_dump_layout(backend_repr)]
type OptMut<'a> = Option<&'a mut String>;
//~^ ERROR: Scalar(pointer is 0..=18446744073709551615)
