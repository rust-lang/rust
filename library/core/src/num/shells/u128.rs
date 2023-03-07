//! Redundant constants module for the [`u128` primitive type][u128].
//!
//! New code should use the associated constants directly on the primitive type.

#![stable(feature = "i128", since = "1.26.0")]
#![deprecated(
    since = "TBD",
    note = "all constants in this module replaced by associated constants on `u128`"
)]

int_module! { u128, #[stable(feature = "i128", since="1.26.0")] }
