#![deny(rustdoc::broken_footnote)]

//! Footnote referenced [^1]. And [^2]. And [^bla].
//!
//! [^1]: footnote defined
//~^^^ ERROR: no footnote definition matching this footnote
//~| ERROR: no footnote definition matching this footnote

//! [^*] special characters can appear within footnote references
//~^ ERROR: no footnote definition matching this footnote
//!
//! [^**]
//!
//! [^**]: not an error
//!
//! [^\_] so can escaped characters
//~^ ERROR: no footnote definition matching this footnote

// Backslash escaped footnotes should not be recognized:
//! [\^4]
//!
//! [^5\]
//!
//! \[^yup]
//!
//! [^foo\
//! bar]
