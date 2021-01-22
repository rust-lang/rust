#![stable(feature = "rust1", since = "1.0.0")]

//! Thread-safe reference-counting pointers.
//!
//! See the [`Arc<T>`][Arc] documentation for more details.

mod arc;
mod weak;

#[stable(feature = "rust1", since = "1.0.0")]
pub use arc::Arc;

#[stable(feature = "rust1", since = "1.0.0")]
pub use weak::Weak;

use arc::ArcInner;
