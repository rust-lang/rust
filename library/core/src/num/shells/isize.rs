//! Constants for the pointer-sized signed integer type.
//!
//! *[See also the `isize` primitive type](../../std/primitive.isize.html).*
//!
//! New code should use the associated constants directly on the primitive type.

#![stable(feature = "rust1", since = "1.0.0")]
#![rustc_deprecated(
    since = "TBD",
    reason = "all constants in this module replaced by associated constants on `isize`"
)]

int_module! { isize }
