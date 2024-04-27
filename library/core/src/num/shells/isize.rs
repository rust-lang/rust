//! Redundant constants module for the [`isize` primitive type][isize].
//!
//! New code should use the associated constants directly on the primitive type.

#![stable(feature = "rust1", since = "1.0.0")]
#![deprecated(
    since = "TBD",
    note = "all constants in this module replaced by associated constants on `isize`"
)]

int_module! { isize }
