//! Redundant constants module for the [`i128` primitive type][i128].
//!
//! New code should use the associated constants directly on the primitive type.

#![stable(feature = "i128", since = "1.26.0")]
#![deprecated(
    since = "TBD",
    note = "all constants in this module replaced by associated constants on `i128`"
)]

int_module! { i128, #[stable(feature = "i128", since="1.26.0")] }
