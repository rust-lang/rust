//! Constants for the pointer-sized unsigned integer type.
//!
//! *[See also the `usize` primitive type](../../std/primitive.usize.html).*
//!
//! New code should use the associated constants directly on the primitive type.

#![stable(feature = "rust1", since = "1.0.0")]
#![rustc_deprecated(
    since = "TBD",
    reason = "all constants in this module replaced by associated constants on `usize`"
)]

int_module! { usize }
