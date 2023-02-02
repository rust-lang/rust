//! Constants for the 16-bit unsigned integer type.
//!
//! *[See also the `u16` primitive type][u16].*
//!
//! New code should use the associated constants directly on the primitive type.

#![stable(feature = "rust1", since = "1.0.0")]
#![deprecated(
    since = "1.69.0",
    note = "all constants in this module replaced by associated constants on `u16`"
)]

int_module! { u16 }
