//! The pointer-sized signed integer type.
//!
//! *[See also the `isize` primitive type](../../std/primitive.isize.html).*

#![stable(feature = "rust1", since = "1.0.0")]

#[cfg(target_pointer_width = "16")]
int_module! { isize, -32768, 32767 }

#[cfg(target_pointer_width = "32")]
int_module! { isize, -2147483648, 2147483647 }

#[cfg(target_pointer_width = "64")]
int_module! { isize, -9223372036854775808, 9223372036854775807 }
