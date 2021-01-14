//! Constants for the 128-bit signed integer type.
//!
//! *[See also the `i128` primitive type](../../std/primitive.i128.html).*
//!
//! New code should use the associated constants directly on the primitive type.

#![stable(feature = "i128", since = "1.26.0")]
#![rustc_deprecated(
    since = "TBD",
    reason = "all constants in this module replaced by associated constants on `i128`"
)]

int_module! { i128, #[stable(feature = "i128", since="1.26.0")] }
