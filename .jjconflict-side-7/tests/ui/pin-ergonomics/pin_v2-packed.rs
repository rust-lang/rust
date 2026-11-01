//! `#[pin_v2]` is not allowed on `#[repr(packed)]` types.
//!
//! Drop glue for a packed type moves an over-aligned field to an aligned location before running
//! its destructor. That move carries along any structurally pinned leaf, so a value handed out as
//! `Pin<&mut _>` would be moved before it is dropped, violating `Pin`'s invariant.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/157011>.

#![feature(pin_ergonomics)]
#![allow(incomplete_features)]

use std::marker::PhantomPinned;

#[pin_v2]
#[repr(packed)]
struct Packed { //~ ERROR `#[pin_v2]` types may not have `#[repr(packed)]`
    field: PhantomPinned,
}

// The generic case from the issue: alignment of `T` is unknown at definition time, so this is
// rejected regardless of how it is later monomorphized.
#[pin_v2]
#[repr(C, packed(4))]
struct PackedN<T>(T);
//~^ ERROR `#[pin_v2]` types may not have `#[repr(packed)]`

#[pin_v2]
#[repr(packed)]
union PackedUnion { //~ ERROR `#[pin_v2]` types may not have `#[repr(packed)]`
    field: (),
}

// Allowed: `#[pin_v2]` without `#[repr(packed)]` still compiles.
#[pin_v2]
#[repr(C)]
struct Unpacked<T>(T);

// Allowed: `#[repr(packed)]` without `#[pin_v2]` is unaffected by the ban.
#[repr(packed)]
struct PackedNoPin(u8);

fn main() {}
