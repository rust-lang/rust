#![unstable(feature = "unicode_internals", issue = "none")]
#![allow(missing_docs)]

pub(crate) mod printable;
mod unicode_data;

/// The version of [Unicode](http://www.unicode.org/) that the Unicode parts of
/// `char` and `str` methods are based on.
///
/// New versions of Unicode are released regularly and subsequently all methods
/// in the standard library depending on Unicode are updated. Therefore the
/// behavior of some `char` and `str` methods and the value of this constant
/// changes over time. This is *not* considered to be a breaking change.
///
/// The version numbering scheme is explained in
/// [Unicode 11.0 or later, Section 3.1 Versions of the Unicode Standard](https://www.unicode.org/versions/Unicode11.0.0/ch03.pdf#page=4).
#[stable(feature = "unicode_version", since = "1.45.0")]
pub const UNICODE_VERSION: (u8, u8, u8) = unicode_data::UNICODE_VERSION;

// For use in liballoc, not re-exported in libstd.
pub mod derived_property {
    pub use super::{Case_Ignorable, Cased};
}

pub use unicode_data::alphabetic::lookup as Alphabetic;
pub use unicode_data::case_ignorable::lookup as Case_Ignorable;
pub use unicode_data::cased::lookup as Cased;
pub use unicode_data::cc::lookup as Cc;
pub use unicode_data::conversions;
pub use unicode_data::grapheme_extend::lookup as Grapheme_Extend;
pub use unicode_data::lowercase::lookup as Lowercase;
pub use unicode_data::n::lookup as N;
pub use unicode_data::uppercase::lookup as Uppercase;
pub use unicode_data::white_space::lookup as White_Space;
