#![stable(feature = "rust1", since = "1.0.0")]

//! Thread-safe reference-counting pointers.
//!
//! See the [`Arc<T>`][Arc] documentation for more details.
//!
//! **Note**: This module is only available on platforms that support atomic
//! loads and stores of pointers. This may be detected at compile time using
//! `#[cfg(target_has_atomic = "ptr")]`.

#[unstable(feature = "unique_rc_arc", issue = "112566")]
pub use crate::rcs::arc::UniqueArc;
#[stable(feature = "rust1", since = "1.0.0")]
pub use crate::rcs::arc::{Arc, Weak};
