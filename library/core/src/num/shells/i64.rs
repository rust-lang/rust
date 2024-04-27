//! Redundant constants module for the [`i64` primitive type][i64].
//!
//! New code should use the associated constants directly on the primitive type.

#![stable(feature = "rust1", since = "1.0.0")]
#![deprecated(
    since = "TBD",
    note = "all constants in this module replaced by associated constants on `i64`"
)]

int_module! { i64 }
