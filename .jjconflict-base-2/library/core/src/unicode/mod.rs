//! Unicode internals used in liballoc and libstd. Not public API.
#![unstable(feature = "unicode_internals", issue = "none")]
#![doc(hidden)]

// for use in alloc, not re-exported in std.
#[rustfmt::skip]
pub use unicode_data::conversions;

#[rustfmt::skip]
pub(crate) use unicode_data::alphabetic::lookup as Alphabetic;
pub(crate) use unicode_data::case_ignorable::lookup as Case_Ignorable;
pub(crate) use unicode_data::cf::lookup as Cf;
pub(crate) use unicode_data::cn_planes_0_3::lookup as Cn_planes_0_3;
pub(crate) use unicode_data::default_ignorable_code_point::lookup as Default_Ignorable_Code_Point;
pub(crate) use unicode_data::grapheme_extend::lookup as Grapheme_Extend;
pub(crate) use unicode_data::lowercase::lookup as Lowercase;
pub(crate) use unicode_data::lt::lookup as Lt;
pub(crate) use unicode_data::n::lookup as N;
pub(crate) use unicode_data::uppercase::lookup as Uppercase;
pub(crate) use unicode_data::white_space::lookup as White_Space;

#[allow(unreachable_pub)]
pub mod unicode_data;

/// The version of [Unicode](https://www.unicode.org/) that the Unicode parts of
/// `char` and `str` methods are based on.
///
/// New versions of Unicode are released regularly and subsequently all methods
/// in the standard library depending on Unicode are updated. Therefore the
/// behavior of some `char` and `str` methods and the value of this constant
/// changes over time, within the boundaries of Unicode's [stability policies].
/// This is *not* considered to be a breaking change.
///
/// [stability policies]: https://www.unicode.org/policies/stability_policy.html
///
/// The version numbering scheme is explained in
/// [Section 3.1 (Version Numbering)] of the Unicode Standard.
///
/// [Section 3.1 (Version Numbering)]: https://www.unicode.org/versions/latest/core-spec/chapter-3/#G49512
pub const UNICODE_VERSION: (u8, u8, u8) = unicode_data::UNICODE_VERSION;
