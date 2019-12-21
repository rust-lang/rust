#![unstable(feature = "unicode_internals", issue = "none")]
#![allow(missing_docs)]

mod bool_trie;
pub(crate) mod printable;
pub(crate) mod tables;
pub(crate) mod version;

// For use in liballoc, not re-exported in libstd.
pub mod derived_property {
    pub use crate::unicode::tables::derived_property::{Case_Ignorable, Cased};
}
pub mod conversions {
    pub use crate::unicode::tables::conversions::{to_lower, to_upper};
}
