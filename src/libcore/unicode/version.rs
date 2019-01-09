/// Represents a Unicode Version.
///
/// See also: <http://www.unicode.org/versions/>
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[unstable(feature = "unicode_version", issue = "49726")]
pub struct UnicodeVersion {
    /// Major version.
    pub major: u32,

    /// Minor version.
    pub minor: u32,

    /// Micro (or Update) version.
    pub micro: u32,

    // Private field to keep struct expandable.
    pub(crate) _priv: (),
}
