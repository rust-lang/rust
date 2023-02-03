//! Redundant constants module for the [`i32` primitive type][i32].
//!
//! New code should use the associated constants directly on the primitive type.

#![stable(feature = "rust1", since = "1.0.0")]
#![deprecated(
    since = "TBD",
    note = "all constants in this module replaced by associated constants on `i32`"
)]

int_module! { i32 }
