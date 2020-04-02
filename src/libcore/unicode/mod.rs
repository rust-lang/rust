#![unstable(feature = "unicode_internals", issue = "none")]
#![allow(missing_docs)]

pub(crate) mod printable;
mod unicode_data;
pub(crate) mod version;

use version::UnicodeVersion;

/// The version of [Unicode](http://www.unicode.org/) that the Unicode parts of
/// `char` and `str` methods are based on.
#[unstable(feature = "unicode_version", issue = "49726")]
pub const UNICODE_VERSION: UnicodeVersion = UnicodeVersion {
    major: unicode_data::UNICODE_VERSION.0,
    minor: unicode_data::UNICODE_VERSION.1,
    micro: unicode_data::UNICODE_VERSION.2,
    _priv: (),
};

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
