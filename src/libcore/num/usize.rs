//! The pointer-sized unsigned integer type.
//!
//! *[See also the `usize` primitive type](../../std/primitive.usize.html).*

#![stable(feature = "rust1", since = "1.0.0")]


#[cfg(target_pointer_width = "16")]
uint_module! { usize, 65535 }

#[cfg(target_pointer_width = "32")]
uint_module! { usize, 4294967295 }

#[cfg(target_pointer_width = "64")]
uint_module! { usize, 18446744073709551615 }
