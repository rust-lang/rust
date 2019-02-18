//! The first version of the prelude of The Rust Standard Library.
//!
//! See the [module-level documentation](../index.html) for more.



#![stable(feature = "rust1", since = "1.0.0")]

// Re-exported core operators
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use crate::marker::{Copy, Send, Sized, Sync, Unpin};
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use crate::ops::{Drop, Fn, FnMut, FnOnce};

// Re-exported functions
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use crate::mem::drop;

// Re-exported types and traits
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use crate::clone::Clone;
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use crate::cmp::{PartialEq, PartialOrd, Eq, Ord};
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use crate::convert::{AsRef, AsMut, Into, From};
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use crate::default::Default;
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use crate::iter::{Iterator, Extend, IntoIterator};
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use crate::iter::{DoubleEndedIterator, ExactSizeIterator};
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use crate::option::Option::{self, Some, None};
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use crate::result::Result::{self, Ok, Err};


// The file so far is equivalent to src/libcore/prelude/v1.rs,
// and below to src/liballoc/prelude.rs.
// Those files are duplicated rather than using glob imports
// because we want docs to show these re-exports as pointing to within `std`.


#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use crate::boxed::Box;
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use crate::borrow::ToOwned;
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use crate::slice::SliceConcatExt;
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use crate::string::{String, ToString};
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use crate::vec::Vec;
