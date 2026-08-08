#![deny(rustdoc::broken_footnote)]

//! Footnote referenced [^1]. And [^2]. And [^bla].
//!
//! [^1]: footnote defined
//~^^^ ERROR: no footnote definition matching this footnote
//~| ERROR: no footnote definition matching this footnote

// Should not lint.
//! foo[^1]
//!
//! ```
//!
//! [^1]: bar
//!
//! ```

// Edge cases from https://pulldown-cmark.github.io/pulldown-cmark/specs/footnotes.html
/// The following are not footnote references:
///
/// \[^a]
///
/// [\^b]
///
/// [^c\]
///
/// [^d
/// e]
///
/// [^f\
/// g]
pub struct NotReferences;

/// The following are not footnote references:
///
/// [^a b]
//~^ ERROR: no footnote definition matching this footnote
///
/// [^1\.2]
//~^ ERROR: no footnote definition matching this footnote
///
/// [^*]
//~^ ERROR: no footnote definition matching this footnote
pub struct EdgeCases;
