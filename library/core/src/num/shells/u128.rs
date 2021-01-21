//! Constants for the 128-bit unsigned integer type.
//!
//! *[See also the `u128` primitive type](../../std/primitive.u128.html).*
//!
//! New code should use the associated constants directly on the primitive type.

#![stable(feature = "i128", since = "1.26.0")]
#![rustc_deprecated(
    since = "TBD",
    reason = "all constants in this module replaced by associated constants on `u128`"
)]

int_module! { u128, #[stable(feature = "i128", since="1.26.0")] }
