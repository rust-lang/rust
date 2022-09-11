//! WASI-specific extensions to general I/O primitives.

#![unstable(feature = "wasi_ext", issue = "71213")]

// NOTE: despite the fact that this module is unstable,
// stable Rust had the capability to access the stable
// re-exported items from os::fd::raw through this
// unstable module.
// In PR #95956 the stability checker was changed to check
// all path segments of an item rather than just the last,
// which caused the aforementioned stable usage to regress
// (see issue #99502).
// As a result, the items in os::fd::raw were given the
// rustc_allowed_through_unstable_modules attribute.
// No regression tests were added to ensure this property,
// as CI is not configured to test wasm32-wasi.
// If this module is stabilized,
// you may want to remove those attributes
// (assuming no other unstable modules need them).
pub use crate::os::fd::raw::*;
